# Integration Guide: External Geometry Support

This guide explains how to integrate `geometry_type = 4` (external vertex files) into IAMReX.

## New Files Added

- `VertexFileReader.H` - Reads `.vertex` marker files
- `WingKinematics.H` - Prescribed flapping kinematics (van Veen 2022)
- `ExternalGeometry.H` - Integration layer and data structures

## Changes Required to DiffusedIB.cpp

### 1. Add includes at top of file

```cpp
#include "ExternalGeometry.H"
```

### 2. Add to ParticleProperties namespace (around line 39)

```cpp
// External geometry support
std::string geometry_file;
Real hinge_x{0.0}, hinge_y{0.0}, hinge_z{0.0};
int do_prescribed_motion{0};
Real kinematics_frequency{600.0};
Real kinematics_stroke_amp{70.0};
Real kinematics_pitch_amp{45.0};
```

### 3. Add input parsing (around line 1279)

After the existing `geometry_type` parsing, add:

```cpp
p_file.query("geometry_file", ParticleProperties::geometry_file);
p_file.query("hinge_x", ParticleProperties::hinge_x);
p_file.query("hinge_y", ParticleProperties::hinge_y);
p_file.query("hinge_z", ParticleProperties::hinge_z);
p_file.query("do_prescribed_motion", ParticleProperties::do_prescribed_motion);
p_file.query("kinematics_frequency", ParticleProperties::kinematics_frequency);
p_file.query("kinematics_stroke_amp", ParticleProperties::kinematics_stroke_amp);
p_file.query("kinematics_pitch_amp", ParticleProperties::kinematics_pitch_amp);
```

### 4. Modify InitParticles (around line 430)

After setting `mKernel.geometry_type`, add handling for type 4:

```cpp
if (mKernel.geometry_type == 4) {
    // External geometry from vertex file
    IAMReX::ExternalGeometryData ext_data;
    ext_data.geometry_file = ParticleProperties::geometry_file;
    ext_data.hinge = amrex::RealVect(
        ParticleProperties::hinge_x,
        ParticleProperties::hinge_y,
        ParticleProperties::hinge_z
    );
    ext_data.do_prescribed_motion = (ParticleProperties::do_prescribed_motion != 0);
    ext_data.kinematics.frequency = ParticleProperties::kinematics_frequency;
    ext_data.kinematics.stroke_amplitude = ParticleProperties::kinematics_stroke_amp;
    ext_data.kinematics.pitch_amplitude = ParticleProperties::kinematics_pitch_amp;

    // Initialize from vertex file
    amrex::RealVect center(x[index], y[index], z[index]);
    IAMReX::InitializeExternalGeometry(ext_data, center, 1.0);

    // Store for later use
    IAMReX::g_external_geometries.push_back(ext_data);

    // Set marker count
    mKernel.ml = ext_data.num_markers;
    mKernel.dv = h * h * h;  // Approximate marker volume

    // Copy positions to phiK/thetaK would need different approach
    // For now, we'll modify InitialWithLargrangianPoints to handle type 4
    if (mKernel.ml > max_largrangian_num) max_largrangian_num = mKernel.ml;
}
```

### 5. Modify InitialWithLargrangianPoints (around line 490)

Replace the existing position computation with geometry-type-aware version:

```cpp
void mParticle::InitialWithLargrangianPoints(const kernel& current_kernel){

    if (verbose) amrex::Print() << "mParticle::InitialWithLargrangianPoints\n";

    // For external geometry (type 4), use stored positions
    if (current_kernel.geometry_type == 4) {
        // Find matching external geometry data
        int ext_idx = -1;
        for (size_t i = 0; i < IAMReX::g_external_geometries.size(); ++i) {
            if (IAMReX::g_external_geometries[i].num_markers == current_kernel.ml) {
                ext_idx = i;
                break;
            }
        }

        if (ext_idx < 0) {
            amrex::Abort("External geometry data not found for kernel");
        }

        auto& ext_data = IAMReX::g_external_geometries[ext_idx];
        const auto* pos_x = ext_data.pos_x.dataPtr();
        const auto* pos_y = ext_data.pos_y.dataPtr();
        const auto* pos_z = ext_data.pos_z.dataPtr();

        for(mParIter pti(*mContainer, LOCAL_LEVEL); pti.isValid(); ++pti){
            const Long np = pti.numParticles();
            if(np == 0) continue;
            auto *particles = pti.GetArrayOfStructs().data();

            amrex::ParallelFor(np, [=]
                AMREX_GPU_DEVICE (int i) noexcept {
                    auto id = particles[i].id();
                    if (id > 0 && id <= ext_data.num_markers) {
                        particles[i].pos(0) = pos_x[id - 1];
                        particles[i].pos(1) = pos_y[id - 1];
                        particles[i].pos(2) = pos_z[id - 1];
                    }
                }
            );
        }
    } else {
        // Original sphere-based positioning
        for(mParIter pti(*mContainer, LOCAL_LEVEL); pti.isValid(); ++pti){
            const Long np = pti.numParticles();
            if(np == 0) continue;
            auto *particles = pti.GetArrayOfStructs().data();

            const auto location = current_kernel.location;
            const auto radius = current_kernel.radius;
            const auto* phiK = current_kernel.phiK.dataPtr();
            const auto* thetaK = current_kernel.thetaK.dataPtr();

            amrex::ParallelFor(np, [=]
                AMREX_GPU_DEVICE (int i) noexcept {
                    auto id = particles[i].id();
                    particles[i].pos(0) = location[0] + radius * std::sin(thetaK[id - 1]) * std::cos(phiK[id - 1]);
                    particles[i].pos(1) = location[1] + radius * std::sin(thetaK[id - 1]) * std::sin(phiK[id - 1]);
                    particles[i].pos(2) = location[2] + radius * std::cos(thetaK[id - 1]);
                }
            );
        }
    }

    mContainer->Redistribute();
    if (verbose) {
        amrex::Print() << "[particle] : particle num :" << mContainer->TotalNumberOfParticles() << "\n";
        mContainer->WriteAsciiFile(amrex::Concatenate("particle", 1));
    }
}
```

### 6. Add kinematics update in UpdateParticles

In `UpdateParticles` (around line 800), add kinematics update at the start:

```cpp
// Update prescribed motion for external geometries
for (auto& ext_data : IAMReX::g_external_geometries) {
    if (ext_data.do_prescribed_motion) {
        IAMReX::UpdateExternalGeometryPositions(ext_data, time);
    }
}
```

## Example Input File

```
# Particle input file for flapping wing

x = 0.015
y = 0.015
z = 0.015
rho = 1000.0
velocity_x = 0.0
velocity_y = 0.0
velocity_z = 0.0
omega_x = 0.0
omega_y = 0.0
omega_z = 0.0
TLX = 0
TLY = 0
TLZ = 0
RLX = 0
RLY = 0
RLZ = 0
radius = 0.001
geometry_type = 4

# External geometry parameters
geometry_file = wing.vertex
hinge_x = 0.015
hinge_y = 0.015
hinge_z = 0.015
do_prescribed_motion = 1
kinematics_frequency = 600.0
kinematics_stroke_amp = 70.0
kinematics_pitch_amp = 45.0
```

## Testing

1. Generate vertex file: `uv run generate-wing-planform --shape elliptic --span 3e-3 --chord 1e-3 --spacing 50e-6 --output wing.vertex`
2. Build IAMReX with new files
3. Run with example input file
4. Verify markers initialize at correct positions
5. Verify markers move with kinematics (if enabled)