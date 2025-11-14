// nurbs_interactivo_edicion.cpp
// Compilar (Linux): g++ nurbs_interactivo_edicion.cpp -o nurbs_interactivo_edicion -lGL -lGLU -lglut -std=c++17

#include <GL/glut.h>
#include <GL/glu.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <sstream>
#include <algorithm>

// ---------------------------
// Tipo simple 3D
// ---------------------------
struct Vec3 {
    double x, y, z;
    Vec3(double X = 0, double Y = 0, double Z = 0) : x(X), y(Y), z(Z) {}
    Vec3 operator+(const Vec3& o) const { return { x + o.x, y + o.y, z + o.z }; }
    Vec3 operator*(double s) const { return { x * s, y * s, z * s }; }
};

// ---------------------------
// Configuración NURBS (4x4 por defecto)
// ---------------------------
int degreeU = 3, degreeV = 3;
int nU = 4, nV = 4;

std::vector<std::vector<Vec3>> ctrl = {
    { {0,0,0}, {0,1,2}, {0,2,-1}, {0,3,0} },
    { {1,0,1}, {1,1,3}, {1,2,0},  {1,3,1} },
    { {2,0,0}, {2,1,2}, {2,2,1},  {2,3,0} },
    { {3,0,0}, {3,1,1}, {3,2,2},  {3,3,3} }
};

std::vector<std::vector<double>> weights = {
    {1,1,1,1},
    {1,2,2,1},
    {1,2,2,1},
    {0.1,0.1,0.1,0.1}
};

// Guardamos el estado inicial (para el reset)
std::vector<std::vector<Vec3>> ctrl_initial = ctrl;
std::vector<std::vector<double>> weights_initial = weights;

std::vector<double> knotU;
std::vector<double> knotV;

// ---------------------------
// Utilities: recompute uniform clamped knots
// ---------------------------
void recomputeKnots() {
    knotU.clear(); knotV.clear();
    int mU = nU + degreeU + 1;
    int mV = nV + degreeV + 1;

    // clamped uniform in [0,1]
    // number of internal intervals = nU - degreeU
    for (int i = 0; i < mU; ++i) {
        if (i <= degreeU) knotU.push_back(0.0);
        else if (i >= mU - degreeU - 1) knotU.push_back(1.0);
        else {
            int spans = mU - 2 * (degreeU + 1) + 1;
            int idx = i - degreeU;
            double val = double(idx) / double(spans + 1);
            knotU.push_back(val);
        }
    }

    for (int i = 0; i < mV; ++i) {
        if (i <= degreeV) knotV.push_back(0.0);
        else if (i >= mV - degreeV - 1) knotV.push_back(1.0);
        else {
            int spans = mV - 2 * (degreeV + 1) + 1;
            int idx = i - degreeV;
            double val = double(idx) / double(spans + 1);
            knotV.push_back(val);
        }
    }
}

// inicializar knots al inicio
struct InitKnotsHelper { InitKnotsHelper() { recomputeKnots(); } } initKnotsHelper;

// ---------------------------
// Cox-de-Boor (base B-spline)
// ---------------------------
double basis(int i, int p, double t, const std::vector<double>& k) {
    if (p == 0) {
        if ((k[i] <= t && t < k[i + 1]) || (t == 1.0 && k[i + 1] == 1.0 && k[i] < 1.0))
            return 1.0;
        return 0.0;
    }

    double denomL = k[i + p] - k[i];
    double left = 0.0;
    if (std::abs(denomL) > 1e-12)
        left = (t - k[i]) / denomL * basis(i, p - 1, t, k);

    double denomR = k[i + p + 1] - k[i + 1];
    double right = 0.0;
    if (std::abs(denomR) > 1e-12)
        right = (k[i + p + 1] - t) / denomR * basis(i + 1, p - 1, t, k);

    return left + right;
}

// ---------------------------
// Evaluar superficie NURBS
// ---------------------------
Vec3 evalSurface(double u, double v) {
    Vec3 num(0, 0, 0);
    double den = 0.0;

    for (int i = 0; i < nU; ++i) {
        double Nu = basis(i, degreeU, u, knotU);
        for (int j = 0; j < nV; ++j) {
            double Nv = basis(j, degreeV, v, knotV);
            double B = Nu * Nv * weights[i][j];
            num = num + (ctrl[i][j] * B);
            den += B;
        }
    }
    if (std::abs(den) < 1e-12) return Vec3(0, 0, 0);
    return num * (1.0 / den);
}

// ---------------------------
// Malla (evaluación en rejilla)
// ---------------------------
std::vector<std::vector<Vec3>> makeMesh(int resU = 20, int resV = 20) {
    std::vector<std::vector<Vec3>> mesh(resU, std::vector<Vec3>(resV));
    for (int iu = 0; iu < resU; ++iu) {
        double u = double(iu) / (resU - 1);
        for (int jv = 0; jv < resV; ++jv) {
            double v = double(jv) / (resV - 1);
            mesh[iu][jv] = evalSurface(u, v);
        }
    }
    return mesh;
}

// ---------------------------
// Variables globales de cámara/interacción
// ---------------------------
int windowW = 800, windowH = 600;
double cameraDistance = 10.0;
double cameraFov = 45.0;
double cameraAngleX = 30.0;
double cameraAngleY = 45.0;

int sel_i = -1, sel_j = -1;
std::set<unsigned char> pressedKeys;
std::set<int> pressedSpecialKeys;

// ---------------------------
// Modo texto para entrada de valores (ahora para "X Y Z")
// ---------------------------
bool inputMode = false;
std::string inputBuffer;

// ---------------------------
// Funciones auxiliares HUD
// ---------------------------
void drawText2D(int x, int y, const std::string& s) {
    glRasterPos2i(x, y);
    for (char c : s) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
}

void drawHUD() {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, windowW, 0, windowH);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(1, 1, 1);
    int y = windowH - 18;
    drawText2D(10, y, "Controles: Click=seleccionar | WASD/QE mover | +/- zoom | Flechas rotar");
    y -= 14;
    drawText2D(10, y, "v:editar texto (X Y Z) | Enter=aplicar | Esc=cancelar | p=insert U | o=insert V | i=insert midpoint | R=reset");
    y -= 14;
    if (sel_i >= 0)
        drawText2D(10, y, "Punto seleccionado: [" + std::to_string(sel_i) + "][" + std::to_string(sel_j) + "]");
    else
        drawText2D(10, y, "Punto seleccionado: ninguno");
    y -= 14;
    if (inputMode) {
        drawText2D(10, y, "MODO EDICION: escribe X Y Z (separados por espacios o comas). Enter aplica. Esc cancela.");
        y -= 14;
        drawText2D(10, y, "Input: " + inputBuffer);
    }
    else {
        drawText2D(10, y, "Pulsa 'v' para editar el punto seleccionado numericamente.");
    }

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// ---------------------------
// Selección: clic sobre punto
// ---------------------------
void pickControlPoint(int mx, int my) {
    GLdouble model[16], proj[16];
    GLint view[4];

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluPerspective(cameraFov, (double)windowW / windowH, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glTranslated(0.0, 0.0, -cameraDistance);
    glRotated(cameraAngleX, 1, 0, 0);
    glRotated(cameraAngleY, 0, 1, 0);
    glTranslated(-1.5, -1.5, 0.0);

    glGetDoublev(GL_MODELVIEW_MATRIX, model);
    glGetDoublev(GL_PROJECTION_MATRIX, proj);
    glGetIntegerv(GL_VIEWPORT, view);

    double bestDist = 1e9;
    int bi = -1, bj = -1;

    for (int i = 0; i < nU; ++i) {
        for (int j = 0; j < nV; ++j) {
            GLdouble winx, winy, winz;
            Vec3 p = ctrl[i][j];
            if (gluProject(p.x, p.y, p.z, model, proj, view, &winx, &winy, &winz)) {
                double sx = winx;
                double sy = view[3] - winy;
                double dx = sx - mx;
                double dy = sy - my;
                double d = std::sqrt(dx * dx + dy * dy);
                if (d < bestDist && d < 12.0) {
                    bestDist = d;
                    bi = i; bj = j;
                }
            }
        }
    }

    sel_i = bi; sel_j = bj;
    if (sel_i >= 0) std::cout << "Seleccionado: [" << sel_i << "][" << sel_j << "]\n";
    else std::cout << "No se seleccionó punto.\n";

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// ---------------------------
// Insertar fila/columna interpolando entre vecinos
// ---------------------------
void insertRowAfter(int iAfter) {
    if (iAfter < 0 || iAfter >= nU - 1) {
        std::cout << "No se puede insertar fila fuera de rango.\n";
        return;
    }
    std::vector<std::vector<Vec3>> newCtrl;
    std::vector<std::vector<double>> newW;

    for (int i = 0; i < nU; ++i) {
        newCtrl.push_back(ctrl[i]);
        newW.push_back(weights[i]);
        if (i == iAfter) {
            // crear fila interpolada entre i y i+1
            std::vector<Vec3> row;
            std::vector<double> wrow;
            for (int j = 0; j < nV; ++j) {
                Vec3 a = ctrl[i][j];
                Vec3 b = ctrl[i + 1][j];
                row.push_back(Vec3(0.5 * (a.x + b.x), 0.5 * (a.y + b.y), 0.5 * (a.z + b.z)));
                wrow.push_back(0.5 * (weights[i][j] + weights[i + 1][j]));
            }
            newCtrl.push_back(row);
            newW.push_back(wrow);
        }
    }

    ctrl = newCtrl;
    weights = newW;
    nU = (int)ctrl.size();
    recomputeKnots();
    std::cout << "Insertada fila después de " << iAfter << ". Nuevo nU=" << nU << "\n";
}

void insertColAfter(int jAfter) {
    if (jAfter < 0 || jAfter >= nV - 1) {
        std::cout << "No se puede insertar columna fuera de rango.\n";
        return;
    }
    for (int i = 0; i < nU; ++i) {
        std::vector<Vec3> newRow;
        std::vector<double> newWrow;
        for (int j = 0; j < nV; ++j) {
            newRow.push_back(ctrl[i][j]);
            newWrow.push_back(weights[i][j]);
            if (j == jAfter) {
                Vec3 a = ctrl[i][j];
                Vec3 b = ctrl[i][j + 1];
                newRow.push_back(Vec3(0.5 * (a.x + b.x), 0.5 * (a.y + b.y), 0.5 * (a.z + b.z)));
                newWrow.push_back(0.5 * (weights[i][j] + weights[i][j + 1]));
            }
        }
        ctrl[i] = newRow;
        weights[i] = newWrow;
    }
    nV = (int)ctrl[0].size();
    recomputeKnots();
    std::cout << "Insertada columna después de " << jAfter << ". Nuevo nV=" << nV << "\n";
}

// Inserta midpoint entre el punto seleccionado y su derecho o abajo (prioriza derecho)
void insertMidpointNearSelected() {
    if (sel_i < 0 || sel_j < 0) {
        std::cout << "Selecciona un punto para insertar midpoint.\n";
        return;
    }
    if (sel_j + 1 < nV) {
        // insertar columna después de sel_j
        insertColAfter(sel_j);
        std::cout << "Insertado midpoint entre columnas en fila " << sel_i << ".\n";
    }
    else if (sel_i + 1 < nU) {
        // insertar fila después de sel_i
        insertRowAfter(sel_i);
        std::cout << "Insertado midpoint entre filas en columna " << sel_j << ".\n";
    }
    else {
        std::cout << "No hay vecino derecho ni abajo para insertar midpoint.\n";
    }
}

// ---------------------------
// Dibujo principal
// ---------------------------
void drawScene() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(cameraFov, (double)windowW / windowH, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslated(0.0, 0.0, -cameraDistance);
    glRotated(cameraAngleX, 1, 0, 0);
    glRotated(cameraAngleY, 0, 1, 0);
    glTranslated(-1.5, -1.5, 0.0);

    glPointSize(8.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < nU; ++i)
        for (int j = 0; j < nV; ++j) {
            if (i == sel_i && j == sel_j) glColor3f(0, 1, 0);
            else glColor3f(1, 0, 0);
            Vec3 p = ctrl[i][j];
            glVertex3f((float)p.x, (float)p.y, (float)p.z);
        }
    glEnd();

    // Malla de control
    glColor3f(1, 0.5, 0);
    for (int i = 0; i < nU; ++i) {
        glBegin(GL_LINE_STRIP);
        for (int j = 0; j < nV; ++j)
            glVertex3f((float)ctrl[i][j].x, (float)ctrl[i][j].y, (float)ctrl[i][j].z);
        glEnd();
    }
    for (int j = 0; j < nV; ++j) {
        glBegin(GL_LINE_STRIP);
        for (int i = 0; i < nU; ++i)
            glVertex3f((float)ctrl[i][j].x, (float)ctrl[i][j].y, (float)ctrl[i][j].z);
        glEnd();
    }

    // Superficie
    auto mesh = makeMesh(30, 30);
    glColor3f(0, 0, 1);
    for (size_t iu = 0; iu < mesh.size(); ++iu) {
        glBegin(GL_LINE_STRIP);
        for (size_t jv = 0; jv < mesh[iu].size(); ++jv)
            glVertex3f((float)mesh[iu][jv].x, (float)mesh[iu][jv].y, (float)mesh[iu][jv].z);
        glEnd();
    }
    for (size_t jv = 0; jv < mesh[0].size(); ++jv) {
        glBegin(GL_LINE_STRIP);
        for (size_t iu = 0; iu < mesh.size(); ++iu)
            glVertex3f((float)mesh[iu][jv].x, (float)mesh[iu][jv].y, (float)mesh[iu][jv].z);
        glEnd();
    }

    drawHUD();
    glutSwapBuffers();
}

// ---------------------------
// Callbacks
// ---------------------------
void onDisplay() { drawScene(); }

void onReshape(int w, int h) {
    windowW = w; windowH = h;
    glViewport(0, 0, w, h);
}

void onMouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) pickControlPoint(x, y);
    if ((button == 3 || button == 4) && state == GLUT_UP)
        cameraDistance *= (button == 3 ? 0.9 : 1.1);
    glutPostRedisplay();
}

// Aplica inputBuffer (X Y Z) al punto seleccionado
void applyInputToSelected() {
    if (!inputMode || sel_i < 0 || sel_j < 0) return;
    if (inputBuffer.empty()) return;
    // Reemplazar comas por espacios
    std::string tmp = inputBuffer;
    std::replace(tmp.begin(), tmp.end(), ',', ' ');
    std::istringstream iss(tmp);
    double nx, ny, nz;
    if (iss >> nx >> ny >> nz) {
        ctrl[sel_i][sel_j] = Vec3(nx, ny, nz);
        std::cout << "Aplicado nuevo valor a punto [" << sel_i << "][" << sel_j << "]: "
            << nx << " " << ny << " " << nz << "\n";
    }
    else {
        std::cout << "Entrada inválida. Usa formato: X Y Z (ej: 1.0 2.0 3.0 o 1,0 2,0 3,0)\n";
    }
}

void onKeyboard(unsigned char key, int, int) {
    // Si estamos en modo texto capturamos la entrada
    if (inputMode) {
        if (key == 13 || key == 10) { // Enter
            applyInputToSelected();
            inputMode = false;
            inputBuffer.clear();
        }
        else if (key == 27) { // Esc
            inputMode = false;
            inputBuffer.clear();
            std::cout << "Edicion cancelada.\n";
        }
        else if (key == 8 || key == 127) { // Backspace
            if (!inputBuffer.empty()) inputBuffer.pop_back();
        }
        else {
            // aceptamos dígitos, signos, espacio, coma, punto, e, E, +, -
            char c = (char)key;
            if ((c >= 32 && c <= 126)) {
                inputBuffer.push_back(c);
            }
        }
        glutPostRedisplay();
        return;
    }

    // Si no estamos en modo texto -> teclas normales
    pressedKeys.insert(key);

    if (key == 'r' || key == 'R') {
        ctrl = ctrl_initial;
        weights = weights_initial;
        nU = (int)ctrl.size();
        nV = (int)ctrl[0].size();
        recomputeKnots();
        cameraDistance = 10.0;
        cameraAngleX = 30.0;
        cameraAngleY = 45.0;
        sel_i = sel_j = -1;
        inputMode = false;
        inputBuffer.clear();
        pressedKeys.clear();
        pressedSpecialKeys.clear();
        std::cout << "Reset realizado. Todos los puntos y cámara restaurados.\n";
    }
    else if (key == 'v' || key == 'V') {
        if (sel_i >= 0 && sel_j >= 0) {
            inputMode = true;
            inputBuffer.clear();
            std::cout << "Modo edicion activado. Escribe X Y Z y presiona Enter.\n";
        }
        else {
            std::cout << "No hay punto seleccionado para editar.\n";
        }
    }
    else if (key == 'p' || key == 'P') {
        if (sel_i >= 0 && sel_i < nU - 1) insertRowAfter(sel_i);
        else std::cout << "Selecciona una fila válida (no el último) para insertar después.\n";
    }
    else if (key == 'o' || key == 'O') {
        if (sel_j >= 0 && sel_j < nV - 1) insertColAfter(sel_j);
        else std::cout << "Selecciona una columna válida (no la última) para insertar después.\n";
    }
    else if (key == 'i' || key == 'I') {
        insertMidpointNearSelected();
    }

    glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int, int) { pressedKeys.erase(key); }
void onSpecialKey(int key, int, int) { pressedSpecialKeys.insert(key); }
void onSpecialKeyUp(int key, int, int) { pressedSpecialKeys.erase(key); }

void onIdle() {
    double step = 0.01;
    double rotStep = 1.5;

    if (sel_i >= 0 && sel_j >= 0 && !inputMode) {
        if (pressedKeys.count('w')) ctrl[sel_i][sel_j].y += step;
        if (pressedKeys.count('s')) ctrl[sel_i][sel_j].y -= step;
        if (pressedKeys.count('a')) ctrl[sel_i][sel_j].x -= step;
        if (pressedKeys.count('d')) ctrl[sel_i][sel_j].x += step;
        if (pressedKeys.count('q')) ctrl[sel_i][sel_j].z += step;
        if (pressedKeys.count('e')) ctrl[sel_i][sel_j].z -= step;
    }
    else if (!inputMode) {
        if (pressedKeys.count('+') || pressedKeys.count('=')) cameraDistance *= 0.98;
        if (pressedKeys.count('-')) cameraDistance *= 1.02;
    }

    if (pressedSpecialKeys.count(GLUT_KEY_LEFT))  cameraAngleY -= rotStep;
    if (pressedSpecialKeys.count(GLUT_KEY_RIGHT)) cameraAngleY += rotStep;
    if (pressedSpecialKeys.count(GLUT_KEY_UP))    cameraAngleX -= rotStep;
    if (pressedSpecialKeys.count(GLUT_KEY_DOWN))  cameraAngleX += rotStep;

    glutPostRedisplay();
}

// ---------------------------
// main
// ---------------------------
int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(windowW, windowH);
    glutCreateWindow("NURBS Interactivo - Edit Coordenadas y Insert");

    glEnable(GL_DEPTH_TEST);

    glutDisplayFunc(onDisplay);
    glutReshapeFunc(onReshape);
    glutMouseFunc(onMouse);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutSpecialFunc(onSpecialKey);
    glutSpecialUpFunc(onSpecialKeyUp);
    glutIdleFunc(onIdle);

    std::cout << "=== INSTRUCCIONES ===\n"
        << "Click: seleccionar punto\n"
        << "WASD/QE: mover punto (modo interactivo)\n"
        << "+/-: zoom\n"
        << "Flechas: rotar camara\n"
        << "v: modo edicion texto (escribe X Y Z) - Enter aplica, Esc cancela\n"
        << "p: insertar fila (U) despues de la fila seleccionada\n"
        << "o: insertar columna (V) despues de la columna seleccionada\n"
        << "i: insertar midpoint (prioriza columna derecha, sino fila abajo)\n"
        << "R: reset total\n";

    glutMainLoop();
    return 0;
}

