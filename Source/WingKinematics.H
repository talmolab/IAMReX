/*
 * WingKinematics.H - Prescribed flapping wing kinematics
 *
 * Implements van Veen et al. (2022) sinusoidal kinematics for insect wing motion in the
 * van Veen / Bomphrey axis convention (mosquito-cfd Tier T2a, issue #1):
 *   x = chord (toward trailing edge), y = span (toward tip), z = wing-normal / vertical.
 * Euler rotation R = Rz(phi) * Ry(alpha) * Rx(theta):
 *   stroke phi about the lab vertical z (perpendicular to the span, so the span-tip sweeps),
 *   pitch/angle-of-attack alpha about the span y, deviation theta about the chord x.
 * (van Veen labels stroke gamma and pitch phi; here stroke=phi, pitch=alpha — same composition.)
 *
 * The Python mirror of this rotation is mosquito_cfd.benchmarks.wing_kinematics.rotation_matrix,
 * pinned by test_python_mirror_matches_design_composition; ANY change to this rotation order MUST
 * update that test (and the fork-side conformance test). See docs/coordinate-convention.md.
 *
 * Reference:
 *   van Veen, W.G., van Leeuwen, J.L., & Muijres, F.T. (2022).
 *   "The unsteady aerodynamics of insect wings with rotational stroke accelerations."
 *   J. Fluid Mech., 936, A3. DOI:10.1017/jfm.2022.31
 */

#ifndef WING_KINEMATICS_H
#define WING_KINEMATICS_H

#include <AMReX_RealVect.H>
#include <AMReX_Vector.H>
#include <AMReX_REAL.H>
#include <cmath>

namespace IAMReX {

/**
 * Kinematics parameters for prescribed wing motion.
 */
struct WingKinematicsParams {
    amrex::Real frequency = 600.0;           // Hz
    amrex::Real stroke_amplitude = 70.0;     // degrees (±70° = 140° total)
    amrex::Real pitch_amplitude = 45.0;      // degrees (angle of attack at midstroke)
    amrex::Real deviation_amplitude = 0.0;   // degrees (out-of-plane, typically 0)
    amrex::Real pitch_phase_lead = 90.0;     // degrees (pitch leads stroke by 90°)

    // Derived quantities (computed from above)
    amrex::Real omega() const { return 2.0 * M_PI * frequency; }
    amrex::Real phi_amp_rad() const { return stroke_amplitude * M_PI / 180.0; }
    amrex::Real alpha_amp_rad() const { return pitch_amplitude * M_PI / 180.0; }
    amrex::Real theta_amp_rad() const { return deviation_amplitude * M_PI / 180.0; }
    amrex::Real phase_lead_rad() const { return pitch_phase_lead * M_PI / 180.0; }
};

/**
 * Compute Euler angles at given time.
 *
 * van Veen et al. (2022) Equations 1-2:
 *   φ(t) = φ_amp * sin(ωt)           Stroke angle
 *   α(t) = α_amp * cos(ωt)           Pitch angle (90° phase lead)
 *   θ(t) = θ_amp * sin(2ωt)          Deviation angle (optional, typically 0)
 */
inline void ComputeEulerAngles(
    amrex::Real time,
    const WingKinematicsParams& params,
    amrex::Real& phi,    // stroke angle (rad)
    amrex::Real& alpha,  // pitch angle (rad)
    amrex::Real& theta)  // deviation angle (rad)
{
    amrex::Real omega = params.omega();

    // Stroke: sinusoidal
    phi = params.phi_amp_rad() * std::sin(omega * time);

    // Pitch: cosine (90° phase lead) - this gives advanced rotation
    alpha = params.alpha_amp_rad() * std::cos(omega * time);

    // Deviation: typically zero for planar stroke, or double frequency
    theta = params.theta_amp_rad() * std::sin(2.0 * omega * time);
}

/**
 * Compute 3x3 rotation matrix from Euler angles (van Veen / Bomphrey convention, T2a).
 *
 * Rotation order: R = Rz(phi) * Ry(alpha) * Rx(theta)
 *   stroke phi about lab vertical z, pitch alpha about span y, deviation theta about chord x.
 *
 * This transforms from the wing-fixed (body) frame to the lab frame. Mirrored in Python by
 * mosquito_cfd.benchmarks.wing_kinematics.rotation_matrix (keep in lock-step).
 */
inline void ComputeRotationMatrix(
    amrex::Real phi,    // stroke angle (rad), about lab z
    amrex::Real alpha,  // pitch angle (rad), about span y
    amrex::Real theta,  // deviation angle (rad), about chord x
    amrex::Real R[3][3])
{
    amrex::Real cp = std::cos(phi),   sp = std::sin(phi);
    amrex::Real ca = std::cos(alpha), sa = std::sin(alpha);
    amrex::Real ct = std::cos(theta), st = std::sin(theta);

    // R = Rz(phi) * Ry(alpha) * Rx(theta)
    R[0][0] = cp * ca;
    R[0][1] = cp * sa * st - sp * ct;
    R[0][2] = cp * sa * ct + sp * st;

    R[1][0] = sp * ca;
    R[1][1] = sp * sa * st + cp * ct;
    R[1][2] = sp * sa * ct - cp * st;

    R[2][0] = -sa;
    R[2][1] = ca * st;
    R[2][2] = ca * ct;
}

/**
 * Transform a point by rotation matrix about a pivot point.
 *
 * new_pos = pivot + R * (ref_pos - pivot)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::RealVect TransformPoint(
    const amrex::RealVect& ref_pos,
    const amrex::RealVect& pivot,
    const amrex::Real R[3][3])
{
    // Position relative to pivot
    amrex::Real dx = ref_pos[0] - pivot[0];
    amrex::Real dy = ref_pos[1] - pivot[1];
    amrex::Real dz = ref_pos[2] - pivot[2];

    // Apply rotation
    amrex::RealVect new_pos;
    new_pos[0] = pivot[0] + R[0][0] * dx + R[0][1] * dy + R[0][2] * dz;
    new_pos[1] = pivot[1] + R[1][0] * dx + R[1][1] * dy + R[1][2] * dz;
    new_pos[2] = pivot[2] + R[2][0] * dx + R[2][1] * dy + R[2][2] * dz;

    return new_pos;
}

/**
 * Update all marker positions based on current time.
 *
 * @param time Current simulation time
 * @param reference_positions Original marker positions at t=0
 * @param hinge Wing hinge position (pivot point)
 * @param params Kinematics parameters
 * @param current_positions Output: updated marker positions
 */
inline void UpdateWingPositions(
    amrex::Real time,
    const amrex::Vector<amrex::RealVect>& reference_positions,
    const amrex::RealVect& hinge,
    const WingKinematicsParams& params,
    amrex::Vector<amrex::RealVect>& current_positions)
{
    // Compute current Euler angles
    amrex::Real phi, alpha, theta;
    ComputeEulerAngles(time, params, phi, alpha, theta);

    // Compute rotation matrix
    amrex::Real R[3][3];
    ComputeRotationMatrix(phi, alpha, theta, R);

    // Ensure output vector is sized correctly
    current_positions.resize(reference_positions.size());

    // Transform each marker
    for (size_t i = 0; i < reference_positions.size(); ++i) {
        current_positions[i] = TransformPoint(reference_positions[i], hinge, R);
    }
}

/**
 * Compute marker velocity from position change (finite difference).
 *
 * For prescribed motion, velocity = d(pos)/dt
 * Using central difference: v(t) ≈ (pos(t+dt/2) - pos(t-dt/2)) / dt
 */
inline void ComputeMarkerVelocities(
    amrex::Real time,
    amrex::Real dt,
    const amrex::Vector<amrex::RealVect>& reference_positions,
    const amrex::RealVect& hinge,
    const WingKinematicsParams& params,
    amrex::Vector<amrex::RealVect>& velocities)
{
    amrex::Vector<amrex::RealVect> pos_plus, pos_minus;
    amrex::Real half_dt = 0.5 * dt;

    UpdateWingPositions(time + half_dt, reference_positions, hinge, params, pos_plus);
    UpdateWingPositions(time - half_dt, reference_positions, hinge, params, pos_minus);

    velocities.resize(reference_positions.size());

    for (size_t i = 0; i < reference_positions.size(); ++i) {
        velocities[i] = (pos_plus[i] - pos_minus[i]) / dt;
    }
}

} // namespace IAMReX

#endif // WING_KINEMATICS_H