//
// Created by od641 on 17/11/2025.
//

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "instance.h"
#include "region.h"

static void debug_print(
        compute_t *const *const array, const struct iterator h_bounds, const struct iterator v_bounds,
        FILE *const destination)
{
    for (indexer_t v_idx = v_bounds.begin; v_idx < v_bounds.end; ++v_idx) {
        for (indexer_t h_idx = h_bounds.begin; h_idx < h_bounds.end; ++h_idx)
            fprintf(destination, "%+10lf ", array[h_idx][v_idx]);
        fputc('\n', destination);
    }
}

static void serialise(const struct region *const region, const struct instance *const instance)
{
    FILE * const output_fp = fopen("./out/airfoil.vtr", "w");
    region_serialise_vtk(region, instance, output_fp);
    fclose(output_fp);
}

int main()
{
    const struct instance instance = instance_create();
    struct region region = region_create(&instance);

    instance_describe(&instance, stderr);
    region_describe(&region, stderr);

    region_initialise(&region, &instance);

    static const compute_t max_simulation_runtime = 0.001; // TODO
    static const indexer_t sor_max_iterations = 100;
    static const compute_t sor_residual_epsilon = 0.001;
    static const indexer_t output_freq = 100;

    compute_t simulation_runtime = 0.0;
    indexer_t step_iteration = 0;

    while (simulation_runtime < max_simulation_runtime) {
        // \Delta_t timestep is fixed.
        region_apply_boundary_conditions(&region);
        region_compute_tentative_velocities(&region, &instance);
        region_compute_poisson_source(&region, &instance);

        compute_t residual = INT_MAX;

        for (indexer_t sor_iteration = 0; sor_iteration < sor_max_iterations; ++sor_iteration) {
            // Perform an SOR cycle and halo-exchange the pressure matrix.
            region_sor_cycle(&region, &instance);

#if 0 // TODO
            // Compute the global residual L_2 norm given the cumulative residual and total fluid cell count.
            residual = region_compute_poisson_residual(&region);
            residual = residual / fluid_cell_sum;
#endif

            if (fabs(residual) < sor_residual_epsilon * sor_residual_epsilon)
                break;
        }

        region_update_velocities(&region, &instance);
        simulation_runtime += instance.timestep_duration;

        if (step_iteration % output_freq == 0)
            printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", step_iteration, simulation_runtime, residual);

        ++step_iteration;
    }

    serialise(&region, &instance);

    region_destroy(&region);
    return EXIT_SUCCESS;
}
