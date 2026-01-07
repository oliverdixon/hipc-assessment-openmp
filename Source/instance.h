//
// Created by od641 on 18/11/2025.
//

#ifndef HIPC_ASSESSMENT_INSTANCE_H
#define HIPC_ASSESSMENT_INSTANCE_H

#include <stdio.h>

#include "types.h"

struct region;

struct naca_specifier
{
    unsigned char maximum_camber;
    unsigned char edge_distance;
    unsigned char maximum_thickness;
};

struct instance
{
    const struct compute_dim2 problem_size;
    const compute_t timestep_duration;
    const compute_t sor_omega;
    const struct naca_specifier naca_specifier;
};

struct instance instance_create();

void instance_describe(const struct instance *instance, FILE *destination);

#endif // HIPC_ASSESSMENT_INSTANCE_H
