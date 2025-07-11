#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Función para calcular momentos Hu y firma desde un contorno
void calcularMomentosYFirma(const vector<Point>& contorno, const string& tipoFigura, ofstream& archivo) {
    if (contorno.empty()) {
        cerr << "Error: Contorno vacío para figura " << tipoFigura << endl;
        return;
    }

    // Calcular momentos Hu
    Moments momentos = moments(contorno);
    double hu[7];
    HuMoments(momentos, hu);

    // Calcular centroide
    Point2f centroid(momentos.m10 / momentos.m00, momentos.m01 / momentos.m00);

    // Calcular firma: distancias de puntos al centroide
    vector<float> shapeSignature;
    for (const Point& p : contorno) {
        shapeSignature.push_back(p.x);
        shapeSignature.push_back(p.y);
        // Alternativamente puedes calcular distancias norm(Point2f(p) - centroid)
        // pero aquí replicamos la estructura del ejemplo
    }

    // Guardar en CSV: categoría, Hu moments, firma (x,y alternados)
    if (archivo.is_open()) {
        archivo << tipoFigura;
        for (int i = 0; i < 7; i++) {
            archivo << "," << hu[i];
        }
        for (auto val : shapeSignature) {
            archivo << "," << val;
        }
        archivo << "\n";
    }
}

void processImage(const string& imagePath, const string& label, ofstream& csvFile) {
    Mat image = imread(imagePath);
    if (image.empty()) {
        cerr << "No se pudo cargar la imagen: " << imagePath << endl;
        return;
    }

    Mat grayImage, blurredImage, binaryImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    GaussianBlur(grayImage, blurredImage, Size(5, 5), 0);
    threshold(blurredImage, binaryImage, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    int kernelSize = 7;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(kernelSize, kernelSize));
    Mat closedImage;
    morphologyEx(binaryImage, closedImage, MORPH_CLOSE, kernel, Point(-1, -1), 2);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(closedImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        cerr << "No se encontraron contornos en: " << imagePath << endl;
        return;
    }

    int largestContourIdx = 0;
    double maxArea = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            largestContourIdx = static_cast<int>(i);
        }
    }

    // Aquí usamos directamente el contorno más grande para calcular momentos y firma
    calcularMomentosYFirma(contours[largestContourIdx], label, csvFile);
}

int main() {
    string basePath = "all-images/";
    vector<string> categories = {"circle", "square", "triangle"};

    // Abrir archivo CSV y escribir encabezado
    ofstream csvFile("descriptores.csv");
    csvFile << "Categoria;Hu1;Hu2;Hu3;Hu4;Hu5;Hu6;Hu7;Firma...\n"; // usa ';' si prefieres separador punto y coma

    for (const string& category : categories) {
        string folderPath = basePath + category;

        if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
            cerr << "No se encontró carpeta: " << folderPath << endl;
            continue;
        }

        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (entry.is_regular_file()) {
                processImage(entry.path().string(), category, csvFile);
            }
        }
    }

    csvFile.close();
    cout << "Archivo descriptores.csv generado correctamente." << endl;

    return 0;
}
