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
#define ind2(i, j) ( i + j * (N[0] + 2 * gs))
#define ind3(i, j) ((i + gs) + (j + gs) * (N[0] + 2*gs))

#define relPos(i, j, x_range) ((i + gs) + ((j + gs) * (x_range + 2 * gs)))


#define absPosX(z, x_start, x_range) ((z % (x_range + 2 * gs)) + x_start - gs)
#define absPosY(z, y_start, x_range) ((z / (x_range + 2 * gs)) + y_start - gs)

#define absAbsPosX(z, x_start, x_range) ((z % (x_range)) + x_start)
#define absAbsPosY(z, y_start, y_range) ((z / (y_range)) + y_start)



double c = 0.4; /* Число Куранта. */
double T = 30.0; /* До какого момента времени считаем. */
double K = 2.0; /* Модуль упругости. */
double rho = 1.0; /* Плотность среды. */
int savec = 10; /* Как часто сохранять. */
int m, n; /* Разбиение на квадраты. */

/* Число точек расчетной области по осям. */
int N[2] = {100, 100};

/* Размер расчетной области. */
double H[2] = {100.0, 100.0};

double R = 10.0; /* Размер области возмущения. */

MPI_Datatype phase_type;

static inline double max(double a, double b) { return a > b ? a : b; }
static inline double min(double a, double b) { return a > b ? b : a; }

static inline int processXIndex(rank, count) { return (rank % m); }
static inline int processYIndex(rank, count) { return (rank / m); }

static inline int nextXProcess(rank, count) { return ((processXIndex(rank, count) == m - 1) ? -1 : (rank + 1));  }
static inline int nextYProcess(rank, count) { return ((processYIndex(rank, count) == n - 1) ? -1 : (rank + m));  }
static inline int prevXProcess(rank, count) { return ((processXIndex(rank, count) != 0) ? (rank - 1) : -1); }
static inline int prevYProcess(rank, count) { return ((processYIndex(rank, count) != 0) ? (rank - m) : -1); }

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

/* Тип, задающий границы области вычисления процесса */
typedef struct {
	int startX; /* Левый верхний угол в абсолютных, x */
	int startY; /* Левый верхний угол в абсолютных, y */
	int rangeX; /* Сторона прямоугольника, x */
	int rangeY; /* Сторона прямоугольника, y */
} range_t;

range_t range;

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
	for (j = -gs; j < range.rangeY + gs; j++) {
		for (i = -gs; i < range.rangeX + gs; i++) {
			node_t n = u[relPos(i, j, range.rangeX)];
			Lx(Z0, &n, &u[relPos(i, j, range.rangeX)]);
		}
	}
	for (j = 0; j < range.rangeY; j++) {
		for (i = 0; i < range.rangeX; i++) {
			node_t w;
			w.p  = rusanov3(u[relPos(i + 2, j, range.rangeX)].p, u[relPos(i + 1, j, range.rangeX)].p, u[relPos(i, j, range.rangeX)].p, u[relPos(i - 1, j, range.rangeX)].p);
			w.vx = u[relPos(i, j, range.rangeX)].vx;
			w.vy = rusanov3(u[relPos(i - 2, j, range.rangeX)].vy, u[relPos(i - 1, j, range.rangeX)].vy, u[relPos(i, j, range.rangeX)].vy, u[relPos(i + 1, j, range.rangeX)].vy);
			/* Обратное преобразование. */
			Rx(Z0, &w, &u1[relPos(i, j, range.rangeX)]);
		}
	}
}

void stepy(node_t *u, node_t *u1)
{
	int i, j;
	double Z0 = Z(K, rho);
	/* Преобразуем n -> w. */
	for (j = -gs; j < range.rangeY + gs; j++) {
		for (i = -gs; i < range.rangeX + gs; i++) {
			node_t n = u[relPos(i, j, range.rangeX)];
			Ly(Z0, &n, &u[relPos(i, j, range.rangeX)]);
		}
	}
	for (j = 0; j < range.rangeY; j++) {
		for (i = 0; i < range.rangeX; i++) {
			node_t w;
			w.p = rusanov3(u[relPos(i, j + 2, range.rangeX)].p, u[relPos(i, j + 1, range.rangeX)].p, u[relPos(i, j, range.rangeX)].p, u[relPos(i, j - 1, range.rangeX)].p);
			w.vx = u[relPos(i, j, range.rangeX)].vx;
			w.vy = rusanov3(u[relPos(i, j - 2, range.rangeX)].vy, u[relPos(i, j - 1, range.rangeX)].vy, u[relPos(i, j, range.rangeX)].vy, u[relPos(i, j + 1, range.rangeX)].vy);
			/* Обратное преобразование. */
			Ry(Z0, &w, &u1[relPos(i, j, range.rangeX)]);
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
				//u[ind(i, j)].p = 0.0;
			} else {
				u[ind(i, j)].p = 0.0;
			}
			u[ind(i, j)].vx = u[ind(i, j)].vy = 0.0;
		}
	}
}

void perform_send_results(node_t *u) {
	int i, j, z;
	node_t *send_buf = (node_t*)malloc(sizeof(node_t) * range.rangeX * range.rangeY);

	z = 0;
	for (j = 0; j < range.rangeY; j++) {
		for (i = 0; i < range.rangeX; i++) {
			send_buf[z] = u[relPos(i, j, range.rangeX)];
			z += 1;
		}
	}
	MPI_Send(send_buf, range.rangeX * range.rangeY, phase_type, 0, 0, MPI_COMM_WORLD);
	free(send_buf);
}

/* Вычисляем границы прямоугольника для одного процесса */
range_t get_ranges(int rank, int count) {
	range_t result;
	result.rangeX = N[0] / m;
	result.rangeY = N[1] / n;
	result.startX = result.rangeX * processXIndex(rank, count);
	result.startY = result.rangeY * processYIndex(rank, count) + gs;
	if (nextXProcess(rank, count) == -1) result.rangeX = N[0] - result.startX;
	if (nextYProcess(rank, count) == -1) result.rangeY = N[1] - gs - result.startY;
	return result;
}

void get_dimensions(int count, int *m, int *n) {
	int done = 0;
	int tmpN, tmpM;
	int candidateN, candidateM;
	tmpM = 1;
	tmpN = count;
	candidateM = 1;
	while (!done) {
		if (count % candidateM == 0) {
			candidateN = count / candidateM;
			tmpM = candidateM;
			tmpN = candidateN;
		}
		candidateM += 1;
		if (candidateM >= tmpN) done = 1;
	}
	*m = tmpM;
	*n = tmpN;
}

void get_bounds_x(node_t *u, int rank, int count) {
	int i, j, z;
	MPI_Status st;

	if (prevXProcess(rank, count) != -1) {
		node_t *buf = (node_t*)malloc(sizeof(node_t) * range.rangeY * gs);
		MPI_Recv(buf, range.rangeY * gs, phase_type, prevXProcess(rank, count), 2, MPI_COMM_WORLD, &st);
		z = 0;
		for (j = 0; j < range.rangeY; j++) {
			for (i = -gs; i < 0; i++) {
//				if (j == 0 && i == -gs && rank == 5) printf("%d %d\n", absPosX(z, range.startX, range.rangeX), range.rangeX);
				u[relPos(i, j, range.rangeX)] = buf[z];
				z += 1;
			}
		}
		free(buf);
	}

	if (nextXProcess(rank, count) != -1) {
		node_t *buf = (node_t*)malloc(sizeof(node_t) * range.rangeY * gs);
		MPI_Recv(buf, (range.rangeY * gs), phase_type, nextXProcess(rank, count), 2, MPI_COMM_WORLD, &st);
		z = 0;
		for (j = 0; j < range.rangeY; j++) {
			for (i = range.rangeX; i < range.rangeX + gs; i++) {
				u[relPos(i, j, range.rangeX)] = buf[z];
				z += 1;
			}
		}
		free(buf);
	}
	//printf("it's me, %d!\n", rank);
}

void get_bounds_y(node_t *u, int rank, int count) {
	int i, j, z;
	MPI_Status st;

	if (prevYProcess(rank, count) != -1) {
		node_t *buf = (node_t*)malloc(sizeof(node_t) * range.rangeX * gs);

		MPI_Recv(buf, range.rangeX * gs, phase_type, prevYProcess(rank, count), 1, MPI_COMM_WORLD, &st);
		z = 0;
		for (i = 0; i < range.rangeX; i++) {
			for (j = -gs; j < 0; j++) {
				u[relPos(i, j, range.rangeX)] = buf[z];
				z += 1;
			}
		}
		free(buf);
	}

	if (nextYProcess(rank, count) != -1) {
		node_t *buf = (node_t*)malloc(sizeof(node_t) * range.rangeX * gs);
		MPI_Recv(buf, range.rangeX * gs, phase_type, nextYProcess(rank, count), 1, MPI_COMM_WORLD, &st);
		z = 0;
		for (i = 0; i < range.rangeX; i++) {
			for (j = range.rangeY; j < range.rangeY + gs; j++) {
				u[relPos(i, j, range.rangeX)] = buf[z];
				z += 1;
			}
		}
		free(buf);
	}

}

void send_bounds_y(node_t *u, int rank, int count) {
	int i, j, z;

	if (prevYProcess(rank, count) != -1) {
		node_t *buf = (node_t*)malloc(sizeof(node_t) * range.rangeX * gs);
		

		z = 0;
		for (i = 0; i < range.rangeX; i++) {
			for (j = 0; j < gs; j++) {
				buf[z] = u[relPos(i, j, range.rangeX)];
				z += 1;
			}
		}
		MPI_Send(buf, range.rangeX * gs, phase_type, prevYProcess(rank, count), 1, MPI_COMM_WORLD);
		free(buf);
	}

	if (nextYProcess(rank, count) != -1) {
		node_t *buf = (node_t*)malloc(sizeof(node_t) * range.rangeX * gs);
		
		z = 0;
		for (i = 0; i < range.rangeX; i++) {
			for (j = range.rangeY - gs; j < range.rangeY; j++) {
				buf[z] = u[relPos(i, j, range.rangeX)];
				z += 1;
			}
		}
		MPI_Send(buf, range.rangeX * gs, phase_type, nextYProcess(rank, count), 1, MPI_COMM_WORLD);
		free(buf);
	}
}

void send_bounds_x(node_t *u, int rank, int count) {
	int i, j, z;
	MPI_Status st;

	if (prevXProcess(rank, count) != -1) {

		node_t *buf = (node_t*)malloc(sizeof(node_t) * range.rangeY * gs);

		z = 0;
		for (j = 0; j < range.rangeY; j++) {
			for (i = 0; i < gs; i++) {
				buf[z] = u[relPos(i, j, range.rangeX)];
				z += 1;
			}
		}
		MPI_Send(buf, range.rangeY * gs, phase_type, prevXProcess(rank, count), 2, MPI_COMM_WORLD);
		free(buf);
	}

	if (nextXProcess(rank, count) != -1) {

		node_t *buf = (node_t*)malloc(sizeof(node_t) * range.rangeY * gs);

		z = 0;
		for (j = 0; j < range.rangeY; j++) {
			for (i = range.rangeX - gs; i < range.rangeX; i++) {
				buf[z] = u[relPos(i, j, range.rangeX)];
				z += 1;
			}
		}
		MPI_Send(buf, range.rangeY * gs, phase_type, nextXProcess(rank, count), 2, MPI_COMM_WORLD);

		free(buf);
	}
}

int main(int argc, char **argv)
{
	/* Шаг сетки. */
	double h[2] = {H[0] / N[0], H[1] / N[1]};
	double o[2] = {0.0, 0.0};
	double dt = c * min(h[0], h[1]) / c1(K, rho); /* Шаг интегрирования по времени. */
	int steps = (int)(T / dt);
	node_t *u, *u1, *v1, *v;
	node_t *send_buf;
	int i, j, z, rank, count, k;
	char buf[256];
	const char *save[3] = {"p", "vx", "vy"};
	double t = 0.0;
	MPI_Status st;
	range_t *ranges;
	int send_size;
	int x, y;

	MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &count);

	get_dimensions(count, &m, &n);

	printf("%lf\n", dt);

	MPI_Type_contiguous(3, MPI_DOUBLE, &phase_type);
	MPI_Type_commit(&phase_type);

	range = get_ranges(rank, count);

	if (!rank) {
		//printf("%d %d\n", m, n);
		u =  (node_t*)malloc(sizeof(node_t) * (N[0] + 2 * gs) * (N[1] + 2 * gs));
		//u1 = (node_t*)malloc(sizeof(node_t) * (N[0] + 2 * gs) * (N[1] + 2 * gs));
		init(u, h, o);

		ranges = (range_t*)malloc(sizeof(range_t) * count);

		ranges[0] = range;

		v = (node_t*)malloc(sizeof(node_t) * (range.rangeX + 2 * gs) * (range.rangeY + 2 * gs));
		v1 = (node_t*)malloc(sizeof(node_t) * (range.rangeX + 2 * gs) * (range.rangeY + 2 * gs));

		send_size = (ranges[0].rangeX + (2 * gs)) * (ranges[0].rangeY + (2 * gs));
		for (z = 0; z < send_size; z++) {
			v[z] = u[ind(absPosX(z, ranges[0].startX, ranges[0].rangeX), absPosY(z, ranges[0].startY, ranges[0].rangeX))];
		}

		/* Раздаем куски процессам */
		for (j = 1; j < count; j++) {
			ranges[j] = get_ranges(j, count);
			//printf("rank: %d, %d %d %d %d\n", j, ranges[j].startX, ranges[j].startY, ranges[j].rangeX, ranges[j].rangeY);
			send_size = (ranges[j].rangeX + (2 * gs)) * (ranges[j].rangeY + (2 * gs));
			send_buf = (node_t*)malloc(sizeof(node_t) * send_size);
			for (z = 0; z < send_size; z++) {
				/*node_t n;
				n.p = 1.0;
				if(j == 3) n.p = 0.0;
				n.vx = 0.0;
				n.vy = 0.0;*/
				send_buf[z] = u[ind(absPosX(z, ranges[j].startX, ranges[j].rangeX), absPosY(z, ranges[j].startY, ranges[j].rangeX))];
				//u[ind(absPosX(z, ranges[j].startX, ranges[j].rangeX), absPosY(z, ranges[j].startY, ranges[j].rangeY))] = n;
			}
			MPI_Send(send_buf, send_size, phase_type, j, 0, MPI_COMM_WORLD);
			free(send_buf);
		}
	} else {
		u = (node_t*)malloc(sizeof(node_t) * (range.rangeX + 2 * gs) * (range.rangeY + 2 * gs));
		u1 = (node_t*)malloc(sizeof(node_t) * (range.rangeX + 2 * gs) * (range.rangeY + 2 * gs));

		/* Принимаем начальные куски */
		send_size = (range.rangeX + (2 * gs)) * (range.rangeY + (2 * gs));

		MPI_Recv(u, send_size, phase_type, 0, 0, MPI_COMM_WORLD, &st);
		/*if (processXIndex(rank,count) != m - 1) {
			for (k = 0; k < send_size; k ++) {
				u[k].p = 0.0;
			}
		} else {
			for (k = 0; k < send_size; k ++) {
				u[k].p = 1.0;
			}
		}*/
	}

	for (i = 0; i < steps; i++) {

		/* Принимаем результаты в записывающий процесс */
		if (!rank && i) { // если процесс первый и такт вычисления не нулевой
			z = 0;
			for (j = 0; j < range.rangeY; j++) {
				for (k = 0; k < range.rangeX; k++) {
					u[ind(absAbsPosX(z, ranges[0].startX, ranges[0].rangeX), absAbsPosY(z, ranges[0].startY, ranges[0].rangeX))] = v[relPos(k, j, range.rangeX)];
					z += 1;
				}
			}
			for (j = 1; j < count; j++) {
				send_size = ranges[j].rangeX * ranges[j].rangeY;
				send_buf = (node_t*)malloc(sizeof(node_t) * send_size);
				MPI_Recv(send_buf, send_size, phase_type, j, 0, MPI_COMM_WORLD, &st);
				for (z = 0; z < send_size; z++) {
					u[ind(absAbsPosX(z, ranges[j].startX, ranges[j].rangeX), absAbsPosY(z, ranges[j].startY, ranges[j].rangeX))] = send_buf[z];
				}
				free(send_buf);
			}
		}


		/* Сохраняем посчитанные значения. */
		if (i % savec == 0 && !rank) {
			sprintf(buf, "data_%06d.vtk", i);
			write_to_vtk2d((double*)u, buf, save, N, o, h, gs, 3);
		}

		if (i && rank) {
			get_bounds_x(u, rank, count);
		} else if (i) {
			get_bounds_x(v, rank, count);
		}

		/* Обновляем значение. */
		if (rank) {
				stepx(u, u1);
				send_bounds_y(u1, rank, count);
				get_bounds_y(u1, rank, count);
				stepy(u1, u);
				if (i != steps - 1) send_bounds_x(u, rank, count);
			} else {
				stepx(v, v1);
				send_bounds_y(v1, rank, count);
				get_bounds_y(v1, rank, count);
				stepy(v1, v);
				if (i != steps - 1) send_bounds_x(v, rank, count);
			}
		

		if (rank && i != steps - 1) perform_send_results(u);
		
		/* Счетчик времени. */
		t += dt;
	}
//	printf("%d was here!\n", rank);
	free(u);
	if (rank) free(u1);
	if (!rank) {
		free(v);
		free(v1);
	}
	if (!rank) free(ranges);
	MPI_Finalize();
	return 0;
}
