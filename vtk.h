#ifndef VTK_H
#define VTK_H

#include <stdio.h>

void write_to_vtk(double *u, const char *path, const char *data, const int N, const double o, const double h)
{
	FILE *f;
	int i;
	f = fopen(path, "w");
	fprintf(f, "# vtk DataFile Version 3.0\n");
	fprintf(f, "Created by write_to_vtk\n");
	fprintf(f, "ASCII\n");
	fprintf(f, "DATASET STRUCTURED_POINTS\n");
	fprintf(f, "DIMENSIONS %d 1 1\n", N);
	fprintf(f, "SPACING %f 0.0 0.0\n", h);
	fprintf(f, "ORIGIN %f 0.0 0.0\n", o);
	fprintf(f, "POINT_DATA %d\n", N);
	fprintf(f, "SCALARS %s float 1\n", data);
	fprintf(f, "LOOKUP_TABLE %s_table\n", data);
	for (i = 0; i < N; i++) {
		fprintf(f, "%f\n", u[i]);
	}
	fclose(f);
}

void write_to_vtk2(double *u, const char *path, const char **data, const int N, const double o, const double h, const int k)
{
	FILE *f;
	int i, j;
	f = fopen(path, "w");
	fprintf(f, "# vtk DataFile Version 3.0\n");
	fprintf(f, "Created by write_to_vtk2\n");
	fprintf(f, "ASCII\n");
	fprintf(f, "DATASET STRUCTURED_POINTS\n");
	fprintf(f, "DIMENSIONS %d 1 1\n", N);
	fprintf(f, "SPACING %f 0.0 0.0\n", h);
	fprintf(f, "ORIGIN %f 0.0 0.0\n", o);
	fprintf(f, "POINT_DATA %d\n", N);
	
	for (j = 0; j < k; j++) {
		fprintf(f, "SCALARS %s float 1\n", data[j]);
		fprintf(f, "LOOKUP_TABLE %s_table\n", data[j]);
		for (i = 0; i < N; i++) {
			fprintf(f, "%f\n", u[i * k + j]);
		}
	}
	fclose(f);
}

void write_to_vtk2d(double *u, const char *path, const char **data, const int N[2], const double o[2], const double h[2], const int gs, const int k)
{
	FILE *f;
	int i1, i2, j;
	f = fopen(path, "w");
	fprintf(f, "# vtk DataFile Version 3.0\n");
	fprintf(f, "Created by write_to_vtk2d\n");
	fprintf(f, "ASCII\n");
	fprintf(f, "DATASET STRUCTURED_POINTS\n");
	fprintf(f, "DIMENSIONS %d %d 1\n", N[0], N[1]);
	fprintf(f, "SPACING %f %f 0.0\n", h[0], h[1]);
	fprintf(f, "ORIGIN %f %f 0.0\n", o[0], o[1]);
	fprintf(f, "POINT_DATA %d\n", N[0] * N[1]);
	
	for (j = 0; j < k; j++) {
		fprintf(f, "SCALARS %s float 1\n", data[j]);
		fprintf(f, "LOOKUP_TABLE %s_table\n", data[j]);
		for (i2 = 0; i2 < N[1]; i2++) {
			for (i1 = 0; i1 < N[0]; i1++) {
				fprintf(f, "%f\n", u[((i1 + gs) + (i2 + gs) * N[0]) * k + j]);
			}
		}
	}
	fclose(f);
}


#endif //VTK_H
