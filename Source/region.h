//
// Created by od641 on 18/11/2025.
//

#ifndef HIPC_ASSESSMENT_REGION_H
#define HIPC_ASSESSMENT_REGION_H

#include <stdio.h>

struct instance;

/**
 * Bitwise flags indicating the nature of a cell in the simulation.
 */
enum cell_flags
{
    CELL_BOUNDARY = 0, /**< Boundary cell */

    CELL_FLUID_NORTH = 1, /**< Boundary cell with fluid to the north */
    CELL_FLUID_SOUTH = 1 << 1, /**< Boundary cell with fluid to the south */
    CELL_FLUID_WEST = 1 << 2, /**< Boundary cell with fluid to the west */
    CELL_FLUID_EAST = 1 << 3, /**< Boundary cell with fluid to the east */

    CELL_FLUID_NORTHWEST = CELL_FLUID_NORTH | CELL_FLUID_WEST,
    CELL_FLUID_SOUTHWEST = CELL_FLUID_SOUTH | CELL_FLUID_WEST,
    CELL_FLUID_NORTHEAST = CELL_FLUID_NORTH | CELL_FLUID_EAST,
    CELL_FLUID_SOUTHEAST = CELL_FLUID_SOUTH | CELL_FLUID_EAST,
    CELL_FLUID_ALL = CELL_FLUID_NORTH | CELL_FLUID_SOUTH | CELL_FLUID_EAST | CELL_FLUID_WEST,

    CELL_FLUID = 1 << 4, /**< Fluid cell */
};

/**
 * Derived region parameters used heavily by the iterative solvers.
 */
struct cached_parameters
{
    const compute_t resolution_sq;
};

/**
 * A region describes a spatial object within a simulation containing multiple mutable vector and scalar fields, and
 * their associated metadata.
 */
struct region
{
    compute_t *const *const velocity_x;
    compute_t *const *const velocity_y;
    compute_t *const *const tentative_velocity_x;
    compute_t *const *const tentative_velocity_y;
    compute_t *const *const pressure;
    compute_t *const *const poisson_source;
    enum cell_flags *const *const flags;

    unsigned int fluid_cell_count;

    const struct dim2 exterior_extent;
    const struct dim2 interior_extent;

    const unsigned int resolution;

    const compute_t initial_velocity_x;
    const compute_t initial_velocity_y;
    const compute_t initial_pressure;
    const enum cell_flags initial_flag;

    struct cached_parameters derived_params;
};

/**
 * Create a new region to be managed by the given instance. The new region has initialised metadata and allocated data
 * matrices. To initialise the data to something meaningful, follow this call with @ref region_initialise.
 *
 * @param instance The constructed instance to manage the region.
 * @return The created region
 */
struct region region_create(const struct instance *instance);

/**
 * Reverse heap allocations performed by @ref region_create.
 *
 * @param region The region to destroy.
 */
void region_destroy(struct region *region);

/**
 * Produce a short, human-readable summary of the given region to the given file.
 *
 * @param region The region whose metadata to describe.
 * @param destination The destination stream for the metadata summary text.
 */
void region_describe(const struct region *region, FILE *destination);

/**
 * Apply horizontal fluid flow (in from the west; out to the east), and no-slip boundary conditions on the velocity
 * matrices i.a.w. Eqns. 3.21, 3.22, and 3.33 of Griebel.
 *
 * @param region The region containing the velocity fields to be subject to the boundary conditions.
 */
void region_apply_boundary_conditions(const struct region *region);

/**
 * Initialise the given region's data matrices, trace the airfoil shape specified by the instance into boundary cells,
 * and set the rectangular border.
 *
 * @param region The region to initialise.
 * @param instance The instance by which the region is managed.
 */
void region_initialise(struct region *region, const struct instance *instance);

void region_serialise_vtk(const struct region *region, const struct instance *instance, FILE *destination);

/**
 * Compute the tentative velocities based on self- and cross-advection, and diffusion, from previous velocity values.
 * These are computed i.a.w. Eqns. 3.36 and 3.37 of Griebel.
 *
 * @param region The region containing the previous velocities and current tentative velocities.
 * @param instance The instance managing the region.
 */
void region_compute_tentative_velocities(const struct region *region, const struct instance *instance);

/**
 * Compute the Poisson source/forcing term for the pressure computation i.a.w. Eqn. 3.38 of Griebel. This spans across
 * the entire interior.
 *
 * @param region The region containing the tentative velocities and Poisson targets.
 * @param instance The instance managing the region.
 */
void region_compute_poisson_source(const struct region *region, const struct instance *instance);

/**
 * Perform a single cycle of Successive Overrelaxation i.a.w. Eqn. 3.44 of Griebel to produce revised a revised pressure
 * scalar field.
 *
 * @param region The region containing the pressure matrix on which SOR should be performed.
 * @param instance The instance managing the region.
 */
void region_sor_cycle(const struct region *region, const struct instance *instance);

/**
 * Compute the discrete residual of the Poisson equation i.a.w. Eqns. 3.45 and 3.46 of Griebel.
 *
 * @param region The region containing the populated Poisson scalar field.
 * @return The partial Poisson residual. To compute the L^2 norm, take the square root.
 */
compute_t region_compute_poisson_residual(const struct region * region);

/**
 * Update the X and Y velocities by the tentative velocities and pressures, according to Eqns. 3.34 and 3.35 of Griebel.
 *
 * @param region The region containing the velocities.
 * @param instance The instance managing the target region.
 */
void region_update_velocities(const struct region *region, const struct instance *instance);

#endif // HIPC_ASSESSMENT_REGION_H
