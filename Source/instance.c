//
// Created by od641 on 18/11/2025.
//

#include <omp.h>

#include "instance.h"
#include "region.h"

struct instance instance_create()
{
    const struct instance instance = {
        .problem_size.x = 4.0,
        .problem_size.y = 1.0,

        .timestep_duration = 0.003,
        .sor_omega = 1.7, // See pg. 37 of Griebel for discussion of choosing this.
        .naca_specifier = {
            .maximum_camber = 2,
            .edge_distance = 4,
            .maximum_thickness = 12
        }
    };

    return instance;
}

void instance_describe(const struct instance *instance, FILE *const destination)
{
    fprintf(destination,
            "Instance statistics:\n\t"
            "Global problem size: (%lf, %lf)\n\t"
            "NACA specifier: %2d%1d%1d\n\t"
            "OMP maximum thread count: %d\n",

            instance->problem_size.x, instance->problem_size.y,
            instance->naca_specifier.maximum_camber, instance->naca_specifier.edge_distance,
            instance->naca_specifier.maximum_thickness, omp_get_max_threads());
}
