//
// Created by od641 on 18/11/2025.
//

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <strings.h>

#include "instance.h"
#include "region.h"

enum cell_tags
{
    TAGS_NORTH,
    TAGS_SOUTH,
    TAGS_EAST,
    TAGS_WEST,
    TAGS_SELF,
};

enum sor_phase
{
    SOR_RED = 0,
    SOR_BLACK = 1
};

static compute_t **alloc_2d_compute_array(const struct dim2 size)
{
    compute_t **array = malloc(size.x * sizeof(compute_t *));
    array[0] = calloc(size.x * size.y, sizeof(compute_t));

    for (size_t column_idx = 1; column_idx < size.x; ++column_idx)
        array[column_idx] = &array[0][column_idx * size.y];

    return array;
}

static enum cell_flags **alloc_2d_flags_array(const struct dim2 size)
{
    enum cell_flags **array = malloc(size.x * sizeof(enum cell_flags *));
    array[0] = calloc(size.x * size.y, sizeof(enum cell_flags));

    for (size_t column_idx = 1; column_idx < size.x; ++column_idx)
        array[column_idx] = &array[0][column_idx * size.y];

    return array;
}

static void free_2d_array(void **array)
{
    free(array[0]);
    free(array);
}

static struct iterator get_initial_v_idx_boundaries(
        const struct region *const region, const compute_t problem_height, const float maximum_camber,
        const float edge_distance, const float thickness, const indexer_t h_idx)
{
    struct iterator boundaries = {
        .begin = 0,
        .end = 0
    };

    /*
     * Position along chord, normalised to [0, 1]. From here, 'x' is translated into the co-ordinate space of the
     * global problem, and not the region.
     */
    const compute_t x = (compute_t) h_idx / (compute_t) region->resolution - 0.5f;

    if (x < 0.0 || x > 1.0)
        return boundaries;

    /*
     * The midline distance is the half-thickness from the fixed 'x' to the horizontal central line of the airfoil.
     * It is the Euclidean distance from the 'x' co-ordinate to the midline. This is NACA standard formulae.
     */
    const compute_t x_sq = x * x;
    compute_t midline_distance = 5.0 * thickness *
            (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * x_sq + 0.2843 * x * x_sq - 0.1015 * x_sq * x_sq);

    /*
     * Compute the 'y' co-ordinate of the mean camber line, given the fixed 'x' position. This is NACA standard
     * formulae, represented as a piecewise map over 'x' in intervals [0, p] and (p, 1], where 'p' is the edge
     * distance.
     */
    const compute_t mean_camber_line_y = x <= edge_distance
            ? maximum_camber / (edge_distance * edge_distance) * (2.0 * edge_distance * x - x_sq)
            : // 0 <= x <= p
            maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) * // p < x <= 1
                    (1.0 - 2.0 * edge_distance + 2.0 * edge_distance * x - x_sq);

    /*
     * Use standard calculus formulae to find the numerical derivative of the mean camber line 'y' co-ordinate.
     * Thickness is applied perpendicular to the mean camber line. Use standard geometric formulae to compute the 'y'
     * co-ordinates for the upper and lower camber surface lines.
     */
    const compute_t norm = x <= edge_distance
            ? 2.0 * maximum_camber / (edge_distance * edge_distance) * (edge_distance - x)
            : 2.0 * maximum_camber / ((1.0 - edge_distance) * (1.0 - edge_distance)) * (edge_distance - x);

    midline_distance *= cos(atan(norm));

    const compute_t upper_camber_y = mean_camber_line_y + midline_distance;
    const compute_t lower_camber_y = mean_camber_line_y - midline_distance;

    boundaries.begin = floor((lower_camber_y + problem_height / 2.0) * region->resolution);
    boundaries.end = ceil((upper_camber_y + problem_height / 2.0) * region->resolution);
    return boundaries;
}

static void write_initial_extreme_boundaries(const struct region *const region)
{
    enum cell_flags *const *const flags = region->flags;
    const struct dim2 interior_extents = {
        .x = region->extents.x - 1,
        .y = region->extents.y - 1,
    };

    #pragma omp parallel for schedule(static) shared(flags, interior_extents) default(none)
    for (indexer_t h_idx = 0; h_idx < interior_extents.x; ++h_idx) {
        flags[h_idx][0] = CELL_BOUNDARY;
        flags[h_idx][interior_extents.y - 1] = CELL_BOUNDARY;
    }

    #pragma omp parallel for schedule(static) shared(flags, interior_extents) default(none)
    for (indexer_t v_idx = 0; v_idx < interior_extents.y; ++v_idx) {
        flags[0][v_idx] = CELL_BOUNDARY;
        flags[interior_extents.x - 1][v_idx] = CELL_BOUNDARY;
    }
}

/**
 * Set arbitrary values for the non-covered areas of the tentative velocities and pressure matrices, typically following
 * a computation of tentatives and preceding a computation of the pressure Poisson term. See Eqns. 3.41 and 3.42 of
 * Griebel.
 *
 * @param region The region containing the velocity and pressure matrices.
 */
static void fix_tentative_boundaries(const struct region * const region)
{
    compute_t * const * const velocity_x = region->velocity_x;
    compute_t * const * const velocity_y = region->velocity_y;
    compute_t * const * const tentative_velocity_x = region->tentative_velocity_x;
    compute_t * const * const tentative_velocity_y = region->tentative_velocity_y;
    compute_t * const * const pressure = region->pressure;

    const struct dim2 interior_extents = {
        .x = region->extents.x - 1,
        .y = region->extents.y - 1,
    };

    // F_{0, j} and F_{i_{max}, j}
    // p_{0, j} and p_{i_{max}+1, j}
    // for j = 1, ..., j_{max}
    for (indexer_t v_idx = 1; v_idx < interior_extents.y; ++v_idx) {
        tentative_velocity_x[0][v_idx] = velocity_x[0][v_idx];
        tentative_velocity_x[interior_extents.x - 1][v_idx] = velocity_x[interior_extents.x - 1][v_idx];
        pressure[0][v_idx] = pressure[1][v_idx];
        pressure[region->extents.x - 1][v_idx] = pressure[interior_extents.x - 1][v_idx];
    }

    // G_{i, 0} and G_{i, j_{max}}
    // p_{0, i} and p_{i, j_{max}+1}
    // for i = 1, ..., i_{max}
    for (indexer_t h_idx = 1; h_idx < interior_extents.x - 1; ++h_idx) {
        tentative_velocity_y[h_idx][0] = velocity_y[h_idx][0];
        tentative_velocity_y[h_idx][interior_extents.y - 1] = velocity_y[h_idx][interior_extents.y - 1];
        pressure[h_idx][0] = pressure[h_idx][1];
        pressure[h_idx][region->extents.y - 1] = pressure[h_idx][interior_extents.y - 1];
    }
}

/**
 * Perform a single-phased (red or black) cycle of SOR to update the pressure scalar field. Run for multiple passes over
 * different phases for a fully populated field.
 *
 * @param region The region containing the pressure matrix on which SOR should be performed.
 * @param omega The relaxation parameter; see Chapter 8.3 of Stoer, J. & Bulirsch, R. (1980).
 *  Introduction to Numerical Analysis.
 * @param phase The phase of the checkerboard pattern to populate in the pressure matrix.
 */
static void sor_cycle_phase(const struct region *const region, const compute_t omega, const enum sor_phase phase)
{
    const compute_t step_sq = region->derived_params.resolution_sq;
    const struct dim2 interior_extents = {
        .x = region->extents.x - 1,
        .y = region->extents.y - 1,
    };

    #pragma omp for schedule(guided, 8) nowait
    for (indexer_t h_idx = 1; h_idx < interior_extents.x; ++h_idx)
        // Align to correct parity for RB indexing. See Figure 2 of https://arxiv.org/abs/1401.0763.
        for (indexer_t v_idx = 1 + (h_idx + 1 & 1 ^ (indexer_t) phase); v_idx < interior_extents.y; v_idx += 2) {
            compute_t weight;

            // Epsilon parameters indicate whether fluid lies in the cell in the corresponding direction.
            compute_t epsilon[4] = {
                (region->flags[h_idx][v_idx + 1] & CELL_FLUID) >> 4, // North
                (region->flags[h_idx][v_idx - 1] & CELL_FLUID) >> 4, // South
                (region->flags[h_idx + 1][v_idx] & CELL_FLUID) >> 4, // East
                (region->flags[h_idx - 1][v_idx] & CELL_FLUID) >> 4, // West
            };

            const compute_t poisson[5] = {
                region->poisson_source[h_idx][v_idx + 1], // North
                region->poisson_source[h_idx][v_idx - 1], // South
                region->poisson_source[h_idx + 1][v_idx], // East
                region->poisson_source[h_idx - 1][v_idx], // West
                region->poisson_source[h_idx][v_idx],     // Self
            };

            const compute_t pressure[5] = {
                region->pressure[h_idx][v_idx + 1], // North
                region->pressure[h_idx][v_idx - 1], // South
                region->pressure[h_idx + 1][v_idx], // East
                region->pressure[h_idx - 1][v_idx], // West
                region->pressure[h_idx][v_idx],     // Self
            };

            if (region->flags[h_idx][v_idx] & CELL_FLUID)

                weight = omega / ((epsilon[TAGS_EAST] + epsilon[TAGS_WEST] + epsilon[TAGS_NORTH] + epsilon[TAGS_SOUTH])
                    * step_sq);

            else {

                epsilon[TAGS_EAST] = 0.0;
                epsilon[TAGS_WEST] = pressure[TAGS_WEST] == 0.0 ?
                    0.0 : omega * pressure[TAGS_SELF] / pressure[TAGS_WEST] / step_sq;
                epsilon[TAGS_SOUTH] = 0.0;
                epsilon[TAGS_NORTH] = pressure[TAGS_NORTH] == 0.0 ?
                    0.0 : poisson[TAGS_SELF] / pressure[TAGS_NORTH] / step_sq;

                weight = 1.0;

            }

            const compute_t x_spatial = epsilon[TAGS_EAST] * pressure[TAGS_EAST] +
                    epsilon[TAGS_WEST] * pressure[TAGS_WEST];

            const compute_t y_spatial = epsilon[TAGS_NORTH] * pressure[TAGS_NORTH] +
                epsilon[TAGS_SOUTH] * pressure[TAGS_SOUTH];

            region->pressure[h_idx][v_idx] = (1 - omega) * pressure[TAGS_SELF] + weight *
                (step_sq * (x_spatial + y_spatial) - poisson[TAGS_SELF]);
        }
}

struct region region_create(const struct instance *const instance)
{
    // Number of cells per unit-distance.
    static const unsigned int resolution = 256;

    // Scaled spatial dimensions by the resolution, to map problem size to problem cell counts.
    const struct dim2 cell_counts = {
        .x = (indexer_t) ceil((compute_t) resolution * instance->problem_size.x) + 2,
        .y = (indexer_t) ceil((compute_t) resolution * instance->problem_size.y) + 2
    };

    struct region region = {
        .velocity_x = alloc_2d_compute_array(cell_counts),
        .velocity_y = alloc_2d_compute_array(cell_counts),
        .tentative_velocity_x = alloc_2d_compute_array(cell_counts),
        .tentative_velocity_y = alloc_2d_compute_array(cell_counts),
        .pressure = alloc_2d_compute_array(cell_counts),
        .poisson_source = alloc_2d_compute_array(cell_counts),
        .flags = alloc_2d_flags_array(cell_counts),

        .resolution = resolution,
        .extents = cell_counts,

        .initial_velocity_x = 1.0,
        .initial_velocity_y = 0.0,
        .initial_pressure = 0.0,
        .initial_flag = CELL_FLUID,

        // Store values commonly used in solver loops, dependent on the region parameters.
        .derived_params = {
            .resolution_sq = resolution * resolution
        }
    };

    // All interior cells are initially fluid. This count can be decremented throughout the simulation.
    region.fluid_cell_count = region.extents.x * region.extents.y;

    return region;
}

void region_destroy(const struct region *const region)
{
    free_2d_array((void **) region->velocity_x);
    free_2d_array((void **) region->velocity_y);
    free_2d_array((void **) region->tentative_velocity_x);
    free_2d_array((void **) region->tentative_velocity_y);
    free_2d_array((void **) region->pressure);
    free_2d_array((void **) region->poisson_source);
    free_2d_array((void **) region->flags);
}

void region_apply_boundary_conditions(const struct region *const region)
{
    compute_t *const *const velocity_x = region->velocity_x;
    compute_t *const *const velocity_y = region->velocity_y;
    enum cell_flags *const *const flags = region->flags;
    const struct dim2 interior_extents = {
        .x = region->extents.x - 1,
        .y = region->extents.y - 1,
    };

    #pragma omp parallel for schedule(static) shared(velocity_x, velocity_y, region) default(none)

    for (indexer_t v_idx = 0; v_idx < region->extents.y; ++v_idx) {
        // Fluid freely flows in from the west
        velocity_x[0][v_idx] = velocity_x[1][v_idx];
        velocity_y[0][v_idx] = velocity_y[1][v_idx];

        // Fluid freely flows out to the east
        velocity_x[region->extents.x - 2][v_idx] = velocity_x[region->extents.x - 3][v_idx];
        velocity_y[region->extents.x - 1][v_idx] = velocity_y[region->extents.x - 2][v_idx];
    }

    #pragma omp parallel for schedule(static) shared(velocity_x, velocity_y, region, interior_extents) default(none)

    for (indexer_t h_idx = 0; h_idx < interior_extents.x; ++h_idx) {
        /*
         * The vertical velocity approaches zero at the north and south boundaries, but fluid flows freely in the
         * horizontal direction. */
        velocity_y[h_idx][region->extents.y - 2] = 0.0;
        velocity_x[h_idx][region->extents.y - 1] = velocity_x[h_idx][region->extents.y - 2];

        velocity_y[h_idx][0] = 0.0;
        velocity_x[h_idx][0] = velocity_x[h_idx][1];
    }

    /*
     * Apply no-slip boundary conditions to cells that are adjacent to internal obstacle cells. This forces the
     * velocities to tend towards zero in these cells. This portion is not parallelised as the number of boundary cells
     * is small, and establishment of boundary conditions requires writes into neighbouring cells.
     */
    for (indexer_t h_idx = 1; h_idx < interior_extents.x; ++h_idx)
        for (indexer_t v_idx = 1; v_idx < interior_extents.y; ++v_idx)
            if (!(flags[h_idx][v_idx] & CELL_FLUID))
                switch (flags[h_idx][v_idx]) {
                case CELL_FLUID_NORTH:
                    velocity_y[h_idx][v_idx] = 0.0;
                    velocity_x[h_idx][v_idx] = -velocity_x[h_idx][v_idx + 1];
                    velocity_x[h_idx - 1][v_idx] = -velocity_x[h_idx - 1][v_idx + 1];
                    break;
                case CELL_FLUID_EAST:
                    velocity_x[h_idx][v_idx] = 0.0;
                    velocity_y[h_idx][v_idx] = -velocity_y[h_idx + 1][v_idx];
                    velocity_y[h_idx][v_idx - 1] = -velocity_y[h_idx + 1][v_idx - 1];
                    break;
                case CELL_FLUID_SOUTH:
                    velocity_y[h_idx][v_idx - 1] = 0.0;
                    velocity_x[h_idx][v_idx] = -velocity_x[h_idx][v_idx - 1];
                    velocity_x[h_idx - 1][v_idx] = -velocity_x[h_idx - 1][v_idx - 1];
                    break;
                case CELL_FLUID_WEST:
                    velocity_x[h_idx - 1][v_idx] = 0.0;
                    velocity_y[h_idx][v_idx] = -velocity_y[h_idx - 1][v_idx];
                    velocity_y[h_idx][v_idx - 1] = -velocity_y[h_idx - 1][v_idx - 1];
                    break;
                case CELL_FLUID_NORTHEAST:
                    velocity_y[h_idx][v_idx] = 0.0;
                    velocity_x[h_idx][v_idx] = 0.0;
                    velocity_y[h_idx][v_idx - 1] = -velocity_y[h_idx + 1][v_idx - 1];
                    velocity_x[h_idx - 1][v_idx] = -velocity_x[h_idx - 1][v_idx + 1];
                    break;
                case CELL_FLUID_SOUTHEAST:
                    velocity_y[h_idx][v_idx - 1] = 0.0;
                    velocity_x[h_idx][v_idx] = 0.0;
                    velocity_y[h_idx][v_idx] = -velocity_y[h_idx + 1][v_idx];
                    velocity_x[h_idx - 1][v_idx] = -velocity_x[h_idx - 1][v_idx - 1];
                    break;
                case CELL_FLUID_SOUTHWEST:
                    velocity_y[h_idx][v_idx - 1] = 0.0;
                    velocity_x[h_idx - 1][v_idx] = 0.0;
                    velocity_y[h_idx][v_idx] = -velocity_y[h_idx - 1][v_idx];
                    velocity_x[h_idx][v_idx] = -velocity_x[h_idx][v_idx - 1];
                    break;
                case CELL_FLUID_NORTHWEST:
                    velocity_y[h_idx][v_idx] = 0.0;
                    velocity_x[h_idx - 1][v_idx] = 0.0;
                    velocity_y[h_idx][v_idx - 1] = -velocity_y[h_idx - 1][v_idx - 1];
                    velocity_x[h_idx][v_idx] = -velocity_x[h_idx][v_idx + 1];
                    break;
                default:;
                }

    /*
     * If we're on a western boundary, fix the western-edge velocities such that there is a continual flow of fluid
     * into the simulation space.
     */
    velocity_y[0][0] = 2 * region->initial_velocity_y - velocity_y[1][0];

    #pragma omp parallel for schedule(static) default(none) shared(velocity_x, velocity_y, region, interior_extents)
    for (indexer_t v_idx = 1; v_idx < interior_extents.y; ++v_idx) {
        velocity_x[0][v_idx] = region->initial_velocity_x;
        velocity_y[0][v_idx] = 2 * region->initial_velocity_y - velocity_y[1][v_idx];
    }
}

void region_update_velocities(const struct region *const region, const struct instance *instance)
{
    /*
     * The pressure differential factors are the constants implied by the discretisation of the momentum equation. They
     * represent fixed-axis grid spacings, warped by the timestep duration, to numerically approximate the next velocity
     * values in terms of the computed pressures.
     */
    const compute_t pressure_diff_factor = instance->timestep_duration * region->resolution;

    #pragma omp parallel default(none) shared(region, pressure_diff_factor)
    {
        // X velocities
        #pragma omp for collapse(2) schedule(static) nowait
        for (indexer_t h_idx = 1; h_idx < region->extents.x - 2; ++h_idx)
            for (indexer_t v_idx = 1; v_idx < region->extents.y - 1; ++v_idx)
                if (region->flags[h_idx][v_idx] & CELL_FLUID && region->flags[h_idx + 1][v_idx] & CELL_FLUID)
                    region->velocity_x[h_idx][v_idx] = region->tentative_velocity_x[h_idx][v_idx] -
                        (region->pressure[h_idx + 1][v_idx] - region->pressure[h_idx][v_idx]) * pressure_diff_factor;

        // Y velocities
        #pragma omp for collapse(2) schedule(static) nowait
        for (indexer_t h_idx = 1; h_idx < region->extents.x - 1; ++h_idx)
            for (indexer_t v_idx = 1; v_idx < region->extents.y - 2; ++v_idx)
                if (region->flags[h_idx][v_idx] & CELL_FLUID && region->flags[h_idx][v_idx + 1] & CELL_FLUID)
                    region->velocity_y[h_idx][v_idx] = region->tentative_velocity_y[h_idx][v_idx] -
                        (region->pressure[h_idx][v_idx + 1] - region->pressure[h_idx][v_idx]) * pressure_diff_factor;
    }
}

compute_t region_get_timestep_interval(const struct region *region, const struct instance *instance)
{
    compute_t x_max = -INFINITY;
    compute_t y_max = -INFINITY;

    const indexer_t x_extent = region->extents.x;
    const indexer_t y_extent = region->extents.y;
    compute_t * const * const velocity_x = region->velocity_x;
    compute_t * const * const velocity_y = region->velocity_y;

    #pragma omp parallel default(none) shared(x_extent, y_extent, x_max, y_max, velocity_x, velocity_y)
    {
        #pragma omp for collapse(2) schedule(static) reduction(max:x_max) nowait
        for (indexer_t h_idx = 0; h_idx < x_extent; ++h_idx)
            for (indexer_t v_idx = 1; v_idx < y_extent; ++v_idx)
                x_max = fmax(fabs(velocity_x[h_idx][v_idx]), x_max);

        #pragma omp for collapse(2) schedule(static) reduction(max:x_max) nowait
        for (indexer_t h_idx = 1; h_idx < x_extent; ++h_idx)
            for (indexer_t v_idx = 0; v_idx < y_extent; ++v_idx)
                y_max = fmax(fabs(velocity_y[h_idx][v_idx]), y_max);
    }

    const compute_t grid_spacing = 1.0 / region->resolution;
    const compute_t cfl_limit = fmin(grid_spacing / x_max, grid_spacing / y_max);
    const compute_t reynolds_delta = 1.0 / (1.0 / (grid_spacing * grid_spacing) +
        1 / (grid_spacing * grid_spacing)) * 500 / 2.0;

    return 0.5 * fmin(cfl_limit, reynolds_delta);
}

void region_compute_tentative_velocities(const struct region *const region, const struct instance *instance)
{
    static const compute_t reynolds = 500.0;
    static const compute_t gamma = 0.9; // Upwind differencing factor in PDE discretisation

    const compute_t quarter_resolution = region->resolution / 4.0;
    const compute_t sq_resolution = region->resolution * region->resolution;

    const struct dim2 interior_extents = {
        .x = region->extents.x - 1,
        .y = region->extents.y - 1,
    };

    #pragma omp parallel default(none) \
        shared(interior_extents, gamma, quarter_resolution, sq_resolution, instance, region, reynolds)
    {
        // Get local copies of pointers to avoid excessive dereferencing in loop.
        compute_t * const * const velocity_x = region->velocity_x;
        compute_t * const * const velocity_y = region->velocity_y;
        enum cell_flags * const * const flags = region->flags;

        // X tentative velocities
        compute_t * const * const tentative_velocity_x = region->tentative_velocity_x;
        #pragma omp for collapse(2) schedule(static) nowait
        for (indexer_t h_idx = 1; h_idx < interior_extents.x - 1; ++h_idx)
            for (indexer_t v_idx = 1; v_idx < interior_extents.y; ++v_idx)
                if (flags[h_idx][v_idx] & CELL_FLUID && flags[h_idx + 1][v_idx] & CELL_FLUID) {

                    const compute_t self_advection_x =
                        (
                            (velocity_x[h_idx][v_idx] + velocity_x[h_idx + 1][v_idx]) *
                            (velocity_x[h_idx][v_idx] + velocity_x[h_idx + 1][v_idx]) +
                            gamma * fabs(velocity_x[h_idx][v_idx] + velocity_x[h_idx + 1][v_idx]) *
                            (velocity_x[h_idx][v_idx] - velocity_x[h_idx + 1][v_idx]) -
                            (velocity_x[h_idx - 1][v_idx] + velocity_x[h_idx][v_idx]) *
                            (velocity_x[h_idx - 1][v_idx] + velocity_x[h_idx][v_idx]) -
                            gamma * fabs(velocity_x[h_idx - 1][v_idx] + velocity_x[h_idx][v_idx]) *
                            (velocity_x[h_idx - 1][v_idx] - velocity_x[h_idx][v_idx])
                        ) * quarter_resolution;

                    const compute_t cross_advection_y =
                        (
                            (velocity_y[h_idx][v_idx] + velocity_y[h_idx + 1][v_idx]) *
                            (velocity_x[h_idx][v_idx] + velocity_x[h_idx][v_idx + 1]) +
                            gamma * fabs(velocity_y[h_idx][v_idx] + velocity_y[h_idx + 1][v_idx]) *
                            (velocity_x[h_idx][v_idx] - velocity_x[h_idx][v_idx + 1]) -
                            (velocity_y[h_idx][v_idx - 1] + velocity_y[h_idx + 1][v_idx - 1]) *
                            (velocity_x[h_idx][v_idx - 1] + velocity_x[h_idx][v_idx]) -
                            gamma * fabs(velocity_y[h_idx][v_idx - 1] +
                                velocity_y[h_idx + 1][v_idx - 1]) *
                            (velocity_x[h_idx][v_idx - 1] - velocity_x[h_idx][v_idx])
                        ) * quarter_resolution;

                    const compute_t diffusion =
                        (
                            velocity_x[h_idx + 1][v_idx] -
                            2.0 * velocity_x[h_idx][v_idx] +
                            velocity_x[h_idx - 1][v_idx] +
                            velocity_x[h_idx][v_idx + 1] -
                            2.0 * velocity_x[h_idx][v_idx] +
                            velocity_x[h_idx][v_idx - 1]
                        ) * sq_resolution;

                    tentative_velocity_x[h_idx][v_idx] = velocity_x[h_idx][v_idx] + instance->timestep_duration *
                        (diffusion / reynolds - self_advection_x - cross_advection_y);

                } else
                    // If both adjacent cells are not fluids, the velocity is unchanged.
                    tentative_velocity_x[h_idx][v_idx] = velocity_x[h_idx][v_idx];

        // Y tentative velocities
        compute_t * const * const tentative_velocity_y = region->tentative_velocity_y;
        #pragma omp for collapse(2) schedule(static) nowait
        for (indexer_t h_idx = 1; h_idx < interior_extents.x; ++h_idx)
            for (indexer_t v_idx = 1; v_idx < interior_extents.y - 1; ++v_idx)
                if (flags[h_idx][v_idx] & CELL_FLUID && flags[h_idx][v_idx + 1] & CELL_FLUID) {

                    const compute_t cross_advection_x =
                        (
                            (velocity_x[h_idx][v_idx] + velocity_x[h_idx][v_idx + 1]) *
                            (velocity_y[h_idx][v_idx] + velocity_y[h_idx + 1][v_idx]) +
                            gamma * fabs(velocity_x[h_idx][v_idx] + velocity_x[h_idx][v_idx + 1]) *
                            (velocity_y[h_idx][v_idx] - velocity_y[h_idx + 1][v_idx]) -
                            (velocity_x[h_idx - 1][v_idx] + velocity_x[h_idx - 1][v_idx + 1]) *
                            (velocity_y[h_idx - 1][v_idx] + velocity_y[h_idx][v_idx]) -
                            gamma * fabs(velocity_x[h_idx - 1][v_idx] +
                                velocity_x[h_idx - 1][v_idx + 1]) *
                            (velocity_y[h_idx - 1][v_idx] - velocity_y[h_idx][v_idx])
                        ) * quarter_resolution;

                    const compute_t self_advection_y =
                        (
                            (velocity_y[h_idx][v_idx] + velocity_y[h_idx][v_idx + 1]) *
                            (velocity_y[h_idx][v_idx] + velocity_y[h_idx][v_idx + 1]) +
                            gamma * fabs(velocity_y[h_idx][v_idx] + velocity_y[h_idx][v_idx + 1]) *
                            (velocity_y[h_idx][v_idx] - velocity_y[h_idx][v_idx + 1]) -
                            (velocity_y[h_idx][v_idx - 1] + velocity_y[h_idx][v_idx]) *
                            (velocity_y[h_idx][v_idx - 1] + velocity_y[h_idx][v_idx]) -
                            gamma * fabs(velocity_y[h_idx][v_idx - 1] + velocity_y[h_idx][v_idx]) *
                            (velocity_y[h_idx][v_idx - 1] - velocity_y[h_idx][v_idx])
                        ) * quarter_resolution;

                    const compute_t diffusion =
                        (
                            velocity_y[h_idx + 1][v_idx] -
                            2.0 * velocity_y[h_idx][v_idx] +
                            velocity_y[h_idx - 1][v_idx] +
                            velocity_y[h_idx][v_idx + 1] -
                            2.0 * velocity_y[h_idx][v_idx] +
                            velocity_y[h_idx][v_idx - 1]
                        ) * sq_resolution;

                    tentative_velocity_y[h_idx][v_idx] = velocity_y[h_idx][v_idx] + instance->timestep_duration *
                        (diffusion / reynolds - cross_advection_x - self_advection_y);

                } else
                    // If both adjacent cells are not fluids, the velocity is unchanged.
                    tentative_velocity_y[h_idx][v_idx] = velocity_y[h_idx][v_idx];
    }
}

void region_compute_poisson_source(const struct region *const region, const struct instance *instance)
{
    fix_tentative_boundaries(region);

    compute_t * const * const tentative_velocity_x = region->tentative_velocity_x;
    compute_t * const * const tentative_velocity_y = region->tentative_velocity_y;
    compute_t * const * const poisson_source = region->poisson_source;
    enum cell_flags * const * const flags = region->flags;

    const struct dim2 interior_extents = {
        .x = region->extents.x - 1,
        .y = region->extents.y - 1,
    };

    #pragma omp parallel for collapse(2) schedule(static) default(none) \
        shared(interior_extents, tentative_velocity_x, tentative_velocity_y, flags, region, poisson_source, instance)

    for (indexer_t h_idx = 1; h_idx < interior_extents.x; ++h_idx)
        for (indexer_t v_idx = 1; v_idx < interior_extents.y; ++v_idx)
            if (flags[h_idx][v_idx] & CELL_FLUID) {
                const compute_t x_tent_vel_diff = (tentative_velocity_x[h_idx][v_idx] -
                    tentative_velocity_x[h_idx - 1][v_idx]) * region->resolution;

                const compute_t y_tent_vel_diff = (tentative_velocity_y[h_idx][v_idx] -
                    tentative_velocity_y[h_idx][v_idx - 1]) * region->resolution;

                poisson_source[h_idx][v_idx] = (x_tent_vel_diff + y_tent_vel_diff) / instance->timestep_duration;
            }
}

void region_sor_cycle(const struct region *const region, const struct instance *const instance)
{
    /*
     * Perform red-black SOR to reduce data dependencies. If tracing the computation over the interior pressure grid,
     * this produces a checkerboard. See https://arxiv.org/abs/1401.0763 for discussion. We also unroll the outermost
     * red-phase black-phase loop to avoid the branch instruction and excessive BP failures.
     */

    const compute_t omega = instance->sor_omega;

    #pragma omp parallel default(none) shared(region, omega)
    {
        sor_cycle_phase(region, omega, SOR_RED);
        #pragma omp barrier
        sor_cycle_phase(region, omega, SOR_BLACK);
    }
}

compute_t region_compute_poisson_residual(const struct region *const region)
{
    const compute_t step_sq = region->derived_params.resolution_sq;
    const struct dim2 interior_extents = {
        .x = region->extents.x - 1,
        .y = region->extents.y - 1,
    };

    compute_t residual = 0.0;

    #pragma omp parallel for collapse(2) reduction(+:residual) schedule(static) default(none) \
        shared(interior_extents, region, step_sq)

    for (indexer_t h_idx = 1; h_idx < interior_extents.x; ++h_idx)
        for (indexer_t v_idx = 1; v_idx < interior_extents.y; ++v_idx)
            if (region->flags[h_idx][v_idx] & CELL_FLUID) {
                const compute_t epsilon_east = !!(region->flags[h_idx + 1][v_idx] & CELL_FLUID);
                const compute_t epsilon_west = !!(region->flags[h_idx - 1][v_idx] & CELL_FLUID);
                const compute_t epsilon_north = !!(region->flags[h_idx][v_idx + 1] & CELL_FLUID);
                const compute_t epsilon_south = !!(region->flags[h_idx][v_idx - 1] & CELL_FLUID);

                const compute_t x_residual = (
                    epsilon_east * (region->pressure[h_idx + 1][v_idx] - region->pressure[h_idx][v_idx]) -
                    epsilon_west * (region->pressure[h_idx][v_idx] - region->pressure[h_idx - 1][v_idx])
                ) * step_sq;

                const compute_t y_residual = (
                    epsilon_north * (region->pressure[h_idx][v_idx + 1] - region->pressure[h_idx][v_idx]) -
                    epsilon_south * (region->pressure[h_idx][v_idx] - region->pressure[h_idx][v_idx - 1])
                ) * step_sq;

                const compute_t add = x_residual + y_residual - region->poisson_source[h_idx][v_idx];
                residual += add * add;
            }

    return residual;
}

void region_initialise(struct region *const region, const struct instance *const instance)
{
    // Populate all cells' information matrices with fixed initial values.
    #pragma omp parallel for collapse(2) schedule(static) shared(region) default(none)

    for (indexer_t h_idx = 0; h_idx < region->extents.x; ++h_idx)
        for (indexer_t v_idx = 0; v_idx < region->extents.y; ++v_idx) {
            region->velocity_x[h_idx][v_idx] = region->initial_velocity_x;
            region->velocity_y[h_idx][v_idx] = region->initial_velocity_y;
            region->pressure[h_idx][v_idx] = region->initial_pressure;
            region->flags[h_idx][v_idx] = region->initial_flag;
        }

    // Transform the NACA digits into the scale expected by the initial boundary calculi.
    const float maximum_camber = (float) instance->naca_specifier.maximum_camber / 100.0f;
    const float edge_distance = (float) instance->naca_specifier.edge_distance / 10.0f;
    const float thickness = (float) instance->naca_specifier.maximum_thickness / 100.0f;

    const struct dim2 interior_extents = {
        .x = region->extents.x - 1,
        .y = region->extents.y - 1,
    };

    #pragma omp parallel for schedule(static) default(none) \
        shared(region, instance, maximum_camber, edge_distance, thickness)

    for (indexer_t h_idx = 0; h_idx < region->extents.x; ++h_idx) {
        // Compute the vertical index boundaries of the airfoil body at the fixed horizontal index.
        const struct iterator v_idx_boundaries = get_initial_v_idx_boundaries(
                region, instance->problem_size.y, maximum_camber, edge_distance, thickness, h_idx);

        // Populate the airfoil body with boundary markers.
        for (indexer_t v_idx = v_idx_boundaries.begin; v_idx < v_idx_boundaries.end; ++v_idx)
            region->flags[h_idx][v_idx] = CELL_BOUNDARY;
    }

    write_initial_extreme_boundaries(region);

    // Mask in additional directional indicator flags for non-fluid cells, describing presence of nearby fluid cells.
    enum cell_flags * const * const flags = region->flags;
    indexer_t fluid_cell_count = region->fluid_cell_count;

    #pragma omp parallel for collapse(2) reduction(-:fluid_cell_count) default(none) shared(flags, interior_extents)

    for (indexer_t h_idx = 1; h_idx < interior_extents.x; ++h_idx)
        for (indexer_t v_idx = 1; v_idx < interior_extents.y; ++v_idx)
            if (!(flags[h_idx][v_idx] & CELL_FLUID)) {
                --fluid_cell_count;

                if (flags[h_idx - 1][v_idx] & CELL_FLUID)
                    flags[h_idx][v_idx] |= CELL_FLUID_WEST;
                if (flags[h_idx + 1][v_idx] & CELL_FLUID)
                    flags[h_idx][v_idx] |= CELL_FLUID_EAST;
                if (flags[h_idx][v_idx - 1] & CELL_FLUID)
                    flags[h_idx][v_idx] |= CELL_FLUID_SOUTH;
                if (flags[h_idx][v_idx + 1] & CELL_FLUID)
                    flags[h_idx][v_idx] |= CELL_FLUID_NORTH;
            }

    region->fluid_cell_count = fluid_cell_count;
}

void region_serialise_vtk(const struct region *const region, const struct instance *const instance,
    FILE *const destination)
{
    // Prologue
    fputs("<?xml version=\"1.0\"?>\n"
          "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n", destination);

    const indexer_t h_pixel_count = region->extents.x - 1;
    const indexer_t v_pixel_count = region->extents.y - 1;

    // Grid consisting of singular piece
    fprintf(destination, "\t<RectilinearGrid WholeExtent=\"0 %d 0 %d 0 0\" GhostLevel=\"0\">\n"
        "\t\t<Piece Extent=\"0 %d 0 %d 0 0\">\n", h_pixel_count, v_pixel_count, h_pixel_count, v_pixel_count);

    // Physical positions of X and Y co-ordinates
    fputs("\t\t\t<Coordinates>\n", destination);

    const compute_t problem_space_width = instance->problem_size.x;
    fprintf(destination,
          "\t\t\t\t<DataArray type=\"Float64\" name=\"X\" format=\"ascii\" RangeMin=\"0\" RangeMax=\"%lf\">\n",
          problem_space_width);
    for (indexer_t h_idx = 0; h_idx <= h_pixel_count; ++h_idx)
        fprintf(destination, "%lf ", problem_space_width / h_pixel_count * h_idx);

    fputs("\n\t\t\t\t</DataArray>\n", destination);

    const compute_t problem_space_height = instance->problem_size.y;
    fprintf(destination,
          "\t\t\t\t<DataArray type=\"Float64\" name=\"Y\" format=\"ascii\" RangeMin=\"0\" RangeMax=\"%lf\">\n",
          problem_space_height);
    for (indexer_t v_idx = 0; v_idx <= v_pixel_count; ++v_idx)
        fprintf(destination, "%lf ", problem_space_height / v_pixel_count * v_idx);

    fputs("\n\t\t\t\t</DataArray>\n"
          "\t\t\t\t<DataArray type=\"Float64\" name=\"Z\" format=\"ascii\">0.0</DataArray>\n"
          "\t\t\t</Coordinates>\n", destination);

    // Velocity vectors
    fputs("\t\t\t<PointData Vectors=\"uv\">\n"
          "\t\t\t\t<DataArray type=\"Float64\" Name=\"uv\" NumberOfComponents=\"3\" format=\"ascii\">\n", destination);

    for (indexer_t v_idx = 0; v_idx <= v_pixel_count; ++v_idx)
        for (indexer_t h_idx = 0; h_idx <= h_pixel_count; ++h_idx)
            fprintf(destination, "%lf %lf 0\n", region->velocity_x[h_idx][v_idx], region->velocity_y[h_idx][v_idx]);

    fputs("\t\t\t\t</DataArray>\n"
          "\t\t\t</PointData>\n", destination);

    // Pressure scalars
    fputs("\t\t\t<CellData Scalars=\"p\">\n"
          "\t\t\t\t<DataArray type=\"Float64\" format=\"ascii\" Name=\"p\">\n", destination);

    for (indexer_t v_idx = 0; v_idx < v_pixel_count; ++v_idx) {
        for (indexer_t h_idx = 0; h_idx < h_pixel_count; ++h_idx)
            fprintf(destination, "%lf ", region->pressure[h_idx][v_idx]);
        fputc('\n', destination);
    }

    fputs("\t\t\t\t</DataArray>\n"
          "\t\t\t</CellData>\n", destination);

    // Epilogue
    fputs("\t\t</Piece>\n"
          "\t</RectilinearGrid>\n"
          "</VTKFile>\n", destination);
}
