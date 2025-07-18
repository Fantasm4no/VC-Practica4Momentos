#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

struct Descriptor {
    string label;
    vector<double> huMoments;
    vector<double> signature;
};

// Distancia euclidiana simple entre dos vectores (puedes mejorarla)
double calcularDistancia(const vector<double>& a, const vector<double>& b) {
    double suma = 0.0;
    size_t minSize = min(a.size(), b.size());
    for (size_t i = 0; i < minSize; i++) {
        suma += pow(a[i] - b[i], 2);
    }
    return sqrt(suma);
}

vector<Descriptor> cargarDescriptores(const string& pathCSV) {
    ifstream archivo(pathCSV);
    vector<Descriptor> descriptores;

    string linea;
    getline(archivo, linea); // Saltar encabezado

    while (getline(archivo, linea)) {
        stringstream ss(linea);
        string celda;
        Descriptor desc;

        // Leer etiqueta
        getline(ss, celda, ',');
        desc.label = celda;

        // Leer momentos Hu (7)
        for (int i = 0; i < 7; ++i) {
            getline(ss, celda, ',');
            desc.huMoments.push_back(stod(celda));
        }

        // Leer firma (x,y,x,y...)
        while (getline(ss, celda, ',')) {
            desc.signature.push_back(stod(celda));
        }

        descriptores.push_back(desc);
    }

    return descriptores;
}

Descriptor procesarImagen(const string& path) {
    Mat image = imread(path);
    Mat gray, blurImg, binary;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurImg, Size(5,5), 0);
    threshold(blurImg, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int idx = 0;
    double maxArea = 0;
    for (int i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            idx = i;
        }
    }

    // Calcula momentos Hu y firma de prueba
    Descriptor test;
    Moments m = moments(contours[idx]);
    double hu[7];
    HuMoments(m, hu);
    for (int i = 0; i < 7; ++i) test.huMoments.push_back(hu[i]);

    Point2f centroid(m.m10 / m.m00, m.m01 / m.m00);
    for (const Point& p : contours[idx]) {
        test.signature.push_back(p.x);
        test.signature.push_back(p.y);
    }

    return test;
}

string clasificar(const Descriptor& prueba, const vector<Descriptor>& base) {
    string mejorEtiqueta = "desconocido";
    double mejorDistancia = 1e9;

    for (const Descriptor& ref : base) {
        double d1 = calcularDistancia(prueba.huMoments, ref.huMoments);
        double d2 = calcularDistancia(prueba.signature, ref.signature);
        double total = d1 + d2;

        if (total < mejorDistancia) {
            mejorDistancia = total;
            mejorEtiqueta = ref.label;
        }
    }

    return mejorEtiqueta;
}

int main() {
    string csvPath = "descriptores.csv";
    string imagenPrueba = "cuadrado.png"; // Imagen de prueba

    vector<Descriptor> base = cargarDescriptores(csvPath);
    Descriptor figuraPrueba = procesarImagen(imagenPrueba);
    string resultado = clasificar(figuraPrueba, base);

    cout << "La imagen de prueba fue clasificada como: " << resultado << endl;
    return 0;
}
