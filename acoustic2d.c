/*
 * Двумерное уравнение акустики.
 * Хохлов Николай, k_h@inbox.ru, 2013.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "vtk.h"

/* Размер "приграничного" слоя. */
#define gs 2

#define ind(i, j) ((i + gs) + (j + gs) * N[0])

double c = 0.4; /* Число Куранта. */
double T = 30.0; /* До какого момента времени считаем. */
double K = 2.0; /* Модуль упругости. */
double rho = 1.0; /* Плотность среды. */
int savec = 10; /* Как часто сохранять. */

/* Число точек расчетной области по осям. */
int N[2] = {100, 100};

/* Размер расчетной области. */
double H[2] = {100.0, 100.0};

double R = 10.0; /* Размер области возмущения. */

double max(double a, double b) { return a > b ? a : b; }
double min(double a, double b) { return a > b ? b : a; }

double minmax(double a, double b, double x)
{
	double minx = min(a, b);
	double maxx = max(a, b);
	return min(maxx, max(minx, x));
}

/* Тип узла. */
typedef struct {
	double p; /* Давление. */
	double vx; /* Скорость. */
	double vy;
} node_t;

/* Скорость звука. */
double c1(const double K, const double rho) { return sqrt(K / rho); }

/* Акустический импеданс. */
double Z(const double K, const double rho) { return rho * c1(K, rho); }

/* Действие матрицы Rx:
 * |-Z 0 Z|
 * | 1 0 1|
 * | 0 1 0|
 */
void Rx(const double Z0, const node_t *w, node_t *n)
{
	n->p  = Z0 * (-w->p + w->vy);
	n->vx = w->p + w->vy;
	n->vy = w->vx;
}

/* Действие матрицы Ry:
 * |-Z 0 Z|
 * | 0 1 0|
 * | 1 0 1|
 */
void Ry(const double Z0, const node_t *w, node_t *n)
{
	n->p  = Z0 * (-w->p + w->vy);
	n->vx = w->vx;
	n->vy = w->p + w->vy;
}

/* Действие матрицы Lx:
 * 1/2 |-1/Z 1 0|
 *     | 0   0 2|
 *     | 1/Z 1 0|
 */
void Lx(const double Z0, const node_t *n, node_t *w)
{
	w->p  = 0.5 * (-n->p / Z0 + n->vx);
	w->vx = w->vy;
	w->vy = 0.5 * (n->p / Z0 + n->vx);
}

/* Действие матрицы Ly:
 * 1/2 |-1/Z 0 1|
 *     | 0   2 0|
 *     | 1/Z 0 1|
 */
void Ly(const double Z0, const node_t *n, node_t *w)
{
	w->p  = 0.5 * (-n->p / Z0 + n->vy);
	w->vx = w->vx;
	w->vy = 0.5 * (n->p / Z0 + n->vy);
}

double rusanov3(const double u_2, const double u_1, const double u0, const double u1)
{
	double d0  = u_1 - u0;
	double d1  = u0  - u1;
	double d_1 = u_2 - u_1;
	double v = u0 + c * (d0 + d1) / 2.0
	              + c * c * (d0 - d1) / 2.0
	              + c * (c * c - 1.0) / 6.0 * (d_1 - 2.0 * d0 + d1);
	return minmax(u0, u_1, v);
}

void stepx(node_t *u, node_t *u1)
{
	int i, j;
	double Z0 = Z(K, rho);
	/* Преобразуем n -> w. */
	for (j = -gs; j < N[1] + gs; j++) {
		for (i = -gs; i < N[0] + gs; i++) {
			node_t n = u[ind(i, j)];
			Lx(Z0, &n, &u[ind(i, j)]);
		}
	}
	for (j = 0; j < N[1]; j++) {
		for (i = 0; i < N[0]; i++) {
			node_t w;
			w.p  = rusanov3(u[ind(i + 2, j)].p, u[ind(i + 1, j)].p, u[ind(i, j)].p, u[ind(i - 1, j)].p);
			w.vx = u[ind(i, j)].vx;
			w.vy = rusanov3(u[ind(i - 2, j)].vy, u[ind(i - 1, j)].vy, u[ind(i, j)].vy, u[ind(i + 1, j)].vy);
			/* Обратное преобразование. */
			Rx(Z0, &w, &u1[ind(i, j)]);
		}
	}
}

void stepy(node_t *u, node_t *u1)
{
	int i, j;
	double Z0 = Z(K, rho);
	/* Преобразуем n -> w. */
	for (j = -gs; j < N[1] + gs; j++) {
		for (i = -gs; i < N[0] + gs; i++) {
			node_t n = u[ind(i, j)];
			Ly(Z0, &n, &u[ind(i, j)]);
		}
	}
	for (j = 0; j < N[1]; j++) {
		for (i = 0; i < N[0]; i++) {
			node_t w;
			w.p = rusanov3(u[ind(i, j + 2)].p, u[ind(i, j + 1)].p, u[ind(i, j)].p, u[ind(i, j - 1)].p);
			w.vx = u[ind(i, j)].vx;
			w.vy = rusanov3(u[ind(i, j - 2)].vy, u[ind(i, j - 1)].vy, u[ind(i, j)].vy, u[ind(i, j + 1)].vy);
			/* Обратное преобразование. */
			Ry(Z0, &w, &u1[ind(i, j)]);
		}
	}
}


void init(node_t *u, const double h[2], const double o[2])
{
	int i, j;
	double c[2] = {o[0] + H[0] / 2, o[1] + H[1] / 2};
	for (j = 0; j < N[1]; j++) {
		for (i = 0; i < N[0]; i++) {
			double x[2] = {h[0] * i + o[0], x[1] = h[1] * j + o[1]};
			double d = (x[0] - c[0]) * (x[0] - c[0]) + (x[1] - c[1]) * (x[1] - c[1]);
			if (sqrt(d) < R) {
				u[ind(i, j)].p = 1.0;
			} else {
				u[ind(i, j)].p = 0.0;
			}
			u[ind(i, j)].vx = u[ind(i, j)].vy = 0.0;
		}
	}
}

int main(int argc, char **argv)
{
	/* Шаг сетки. */
	double h[2] = {H[0] / N[0], H[1] / N[1]};
	double o[2] = {0.0, 0.0};
	double dt = c * min(h[0], h[1]) / c1(K, rho); /* Шаг интегрирования по времени. */
	int steps = (int)(T / dt);
	node_t *u  = (node_t*)malloc(sizeof(node_t) * (N[0] + 2 * gs) * (N[1] + 2 * gs));
	node_t *u1 = (node_t*)malloc(sizeof(node_t) * (N[0] + 2 * gs) * (N[1] + 2 * gs));
	int i;
	char buf[256];
	const char *save[3] = {"p", "vx", "vy"};
	double t = 0.0;
	
	init(u, h, o);
	for (i = 0; i < steps; i++) {
		/* Сохраняем посчитанные значения. */
		if (i % savec == 0) {
			sprintf(buf, "data_%06d.vtk", i);
			write_to_vtk2d((double*)u, buf, save, N, o, h, gs, 3);
		}

		/* Обновляем значение. */
		stepx(u, u1);
		stepy(u1, u);
		
		/* Счетчик времени. */
		t += dt;
	}
	free(u);
	free(u1);
	return 0;
}
