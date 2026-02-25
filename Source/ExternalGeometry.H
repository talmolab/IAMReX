/*
 * ExternalGeometry.H - Support for external geometry files and prescribed kinematics
 *
 * Adds geometry_type = 4 to IAMReX for loading .vertex marker files
 * and applying prescribed flapping motion.
 *
 * Usage in input file:
 *   particle_inputs.geometry_type = 4
 *   particle_inputs.geometry_file = wing.vertex
 *   particle_inputs.hinge_x = 0.015
 *   particle_inputs.hinge_y = 0.015
 *   particle_inputs.hinge_z = 0.015
 *   particle_inputs.do_prescribed_motion = 1
 *   particle_inputs.kinematics_frequency = 600.0
 *   particle_inputs.kinematics_stroke_amp = 70.0
 *   particle_inputs.kinematics_pitch_amp = 45.0
 */

#ifndef EXTERNAL_GEOMETRY_H
#define EXTERNAL_GEOMETRY_H

#include <AMReX_RealVect.H>
#include <AMReX_Vector.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Gpu.H>
#include "VertexFileReader.H"
#include "WingKinematics.H"

namespace IAMReX {

/**
 * Storage for external geometry particles.
 * Extends the kernel concept for geometry_type = 4.
 */
struct ExternalGeometryData {
    std::string geometry_file;
    amrex::RealVect hinge{0.0, 0.0, 0.0};
    amrex::Vector<amrex::RealVect> reference_positions;
    amrex::Vector<amrex::RealVect> current_positions;
    amrex::Gpu::PinnedVector<amrex::Real> pos_x;  // Pinned memory: host-writeable, GPU-readable
    amrex::Gpu::PinnedVector<amrex::Real> pos_y;
    amrex::Gpu::PinnedVector<amrex::Real> pos_z;
    WingKinematicsParams kinematics;
    bool do_prescribed_motion = false;
    int num_markers = 0;
};

/**
 * Global storage for external geometry data.
 * One entry per body with geometry_type = 4.
 */
inline amrex::Vector<ExternalGeometryData> g_external_geometries;

/**
 * Read external geometry parameters from input file.
 */
inline void ReadExternalGeometryParams(
    amrex::ParmParse& pp,
    int body_index,
    ExternalGeometryData& data)
{
    // Geometry file
    std::string prefix = "body_" + std::to_string(body_index) + "_";

    // Try body-specific first, then fall back to global
    if (!pp.query(prefix + "geometry_file", data.geometry_file)) {
        pp.get("geometry_file", data.geometry_file);
    }

    // Hinge position
    amrex::Real hx = 0.0, hy = 0.0, hz = 0.0;
    pp.query("hinge_x", hx);
    pp.query("hinge_y", hy);
    pp.query("hinge_z", hz);
    data.hinge = amrex::RealVect(hx, hy, hz);

    // Prescribed motion
    int do_motion = 0;
    pp.query("do_prescribed_motion", do_motion);
    data.do_prescribed_motion = (do_motion != 0);

    // Kinematics parameters (with van Veen defaults)
    pp.query("kinematics_frequency", data.kinematics.frequency);
    pp.query("kinematics_stroke_amp", data.kinematics.stroke_amplitude);
    pp.query("kinematics_pitch_amp", data.kinematics.pitch_amplitude);
    pp.query("kinematics_deviation_amp", data.kinematics.deviation_amplitude);
    pp.query("kinematics_phase_lead", data.kinematics.pitch_phase_lead);
}

/**
 * Initialize external geometry from vertex file.
 */
inline void InitializeExternalGeometry(
    ExternalGeometryData& data,
    const amrex::RealVect& center,
    amrex::Real scale = 1.0)
{
    // Read vertex file
    data.num_markers = ReadVertexFile(
        data.geometry_file,
        data.reference_positions,
        center,
        scale
    );

    // Initialize current positions to reference
    data.current_positions = data.reference_positions;

    // Fill pinned host arrays (accessible from both CPU and GPU; no explicit copy needed)
    data.pos_x.resize(data.num_markers);
    data.pos_y.resize(data.num_markers);
    data.pos_z.resize(data.num_markers);
    for (int i = 0; i < data.num_markers; ++i) {
        data.pos_x[i] = data.reference_positions[i][0];
        data.pos_y[i] = data.reference_positions[i][1];
        data.pos_z[i] = data.reference_positions[i][2];
    }

    amrex::Print() << "Initialized external geometry with " << data.num_markers
                   << " markers from " << data.geometry_file << "\n";
    amrex::Print() << "  Hinge: (" << data.hinge[0] << ", " << data.hinge[1]
                   << ", " << data.hinge[2] << ")\n";
    amrex::Print() << "  Prescribed motion: " << (data.do_prescribed_motion ? "yes" : "no") << "\n";
    if (data.do_prescribed_motion) {
        amrex::Print() << "  Kinematics: f=" << data.kinematics.frequency << " Hz, "
                       << "phi_amp=" << data.kinematics.stroke_amplitude << " deg, "
                       << "alpha_amp=" << data.kinematics.pitch_amplitude << " deg\n";
    }
}

/**
 * Update external geometry positions based on kinematics.
 * Call this at the beginning of each timestep.
 */
inline void UpdateExternalGeometryPositions(
    ExternalGeometryData& data,
    amrex::Real time)
{
    if (!data.do_prescribed_motion) {
        return;
    }

    // Update positions using kinematics
    UpdateWingPositions(
        time,
        data.reference_positions,
        data.hinge,
        data.kinematics,
        data.current_positions
    );

    // Update pinned host arrays (no explicit copy needed; GPU reads pinned memory directly)
    for (int i = 0; i < data.num_markers; ++i) {
        data.pos_x[i] = data.current_positions[i][0];
        data.pos_y[i] = data.current_positions[i][1];
        data.pos_z[i] = data.current_positions[i][2];
    }
}

/**
 * Get marker positions for particle initialization.
 * For geometry_type = 4, this returns the loaded positions
 * instead of computing from phi/theta angles.
 */
inline void GetExternalMarkerPositions(
    const ExternalGeometryData& data,
    amrex::Vector<amrex::RealVect>& positions)
{
    positions = data.current_positions;
}

} // namespace IAMReX

#endif // EXTERNAL_GEOMETRY_H