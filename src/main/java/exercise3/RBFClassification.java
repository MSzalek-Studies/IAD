package exercise3;

import exercise1.activator.SigmoidActivator;
import org.jfree.data.xy.XYSeries;
import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.vector.dense.BasicVector;
import utils.ErrorChart;
import utils.FileUtils;
import utils.MatrixUtils;

import java.io.IOException;
import java.util.Random;

/**
 * Created by marcinus on 19.03.17.
 */
public class RBFClassification {

    /**
     * m - liczba danych testowych
     * k - liczba neuronow w warstwie ukrytej
     * n - liczba cech
     * c - liczba neuronow w wartswie wyjsciowej
     */
    int m;
    int n;
    int k;
    int c;
    Matrix inputs; //m x n
    Matrix rbfCenters;// k x n
    Matrix rbfValues; // m x k
    Matrix outputWeights; //k x c
    Matrix outputWeightsErrors; //k x c
    Matrix outputs; // m x c
    Matrix outputErrors; //m x c
    Matrix expectedValues; // m x c
    Matrix sigmas; //n x 1

    Vector ids;

    public void performClassification(int iterations, int hiddenNeurons, double learningRate) throws IOException {
        Matrix[] matrices = new FileUtils().loadDataFromSingleFileSupervised("classification_train.txt");
        inputs = matrices[0];
        expectedValues = matrices[1];

        m = inputs.rows();
        n = inputs.columns();
        k = hiddenNeurons;
        c = 1;

        rbfCenters = new Basic2DMatrix(k, n);
        rbfValues = new Basic2DMatrix(m, k);
        outputWeights = MatrixUtils.randomlyInitWeights(k, c);
        outputs = new Basic2DMatrix(m, c);
        ids = new BasicVector(m);

        initRandomCenters();
        initSigmas();

        ErrorChart errorChart = new ErrorChart();
        XYSeries series = new XYSeries("Seria");

        for (int i = 0; i < iterations; i++) {
            updateIds();
            updateNeurons();
        }
        initSigmas();

        for (int it = 0; it < iterations; it++) {
            propagateForwards();

            outputErrors = outputs.subtract(expectedValues);
            outputWeightsErrors = rbfValues.transpose().multiply(outputErrors);

            Matrix difference = outputWeightsErrors.multiply(learningRate);
            outputWeights = outputWeights.subtract(difference);

            double cost = outputErrors.transform((i, j, value) -> Math.abs(value)).sum();
            series.add(it, cost);
        }
        errorChart.addSeries(series);
        errorChart.generateChart("classError.png");
    }

    private void propagateForwards() {
        for (int i = 0; i < m; i++) {
            for (int centroidIndex = 0; centroidIndex < k; centroidIndex++) {
                double wartosc = 0;
                for (int cechaIndex = 0; cechaIndex < c; cechaIndex++) {
                    double distanceDlaCechy = Math.pow(inputs.get(i, cechaIndex) - rbfCenters.get(centroidIndex, cechaIndex), 2);
                    wartosc += distanceDlaCechy / (2 * sigmas.get(cechaIndex, 0) * sigmas.get(cechaIndex, 0));
                }
                wartosc = Math.exp(-wartosc);
                rbfValues.set(i, centroidIndex, wartosc);
            }
            //System.out.println(rbfValues.getRow(i));
            outputs = new SigmoidActivator().activate(rbfValues.multiply(outputWeights)).add(1);
        }
    }

    private void initSigmas() {
        sigmas = new Basic2DMatrix(n, 1);
        for (int i = 0; i < n; i++) {
            sigmas.set(i, 0, 2 * rbfCenters.getColumn(i).sum() / (1.0 * k));
        }
    }

    private void initRandomCenters() {
        for (int i = 0; i < k; i++) {
            int row = new Random().nextInt(m);
            rbfCenters.setRow(i, inputs.getRow(row));
        }
    }

    private void predict(Matrix input) {
        inputs = input;
        m = inputs.rows();

        rbfValues = new Basic2DMatrix(m, k);
        outputs = new Basic2DMatrix(m, c);

        propagateForwards();
    }

    private double test(Matrix input, Matrix expected) {
        predict(input);
        int goodClassifications = 0;
        for (int i = 0; i < input.rows(); i++) {
            int expectedResult = (int) expected.get(i, 0);
            int expectedIndex = expectedResult - 1;
            if (outputs.get(i, expectedIndex) == expected.maxInRow(i)) {
                goodClassifications++;
            } else {
                /*System.out.println(input.getRow(i));
                System.out.println(expectedResult);
                System.out.println(predictions.getRow(i) + "\n");*/
            }
        }
        return (double) goodClassifications / (double) input.rows();
    }

    private void updateNeurons() {
        for (int i = 0; i < k; i++) {
            Vector means = new BasicVector(inputs.columns());
            int counter = 0;
            for (int j = 0; j < ids.length(); j++) {
                if (ids.get(j) == i) {
                    Vector inputsRow = inputs.getRow(j);
                    means = means.add(inputsRow);
                    counter++;
                }
            }
            if (counter > 0) {
                means.divide(counter);
                rbfCenters.setRow(i, means);
            }
        }
    }

    private void updateIds() {
        for (int i = 0; i < inputs.rows(); i++) {
            double minDistance = -1;
            for (int j = 0; j < k; j++) {
                double distance = distance(rbfCenters.getRow(j), inputs.getRow(i));
                if (minDistance == -1 || distance < minDistance) {
                    minDistance = distance;
                    ids.set(i, j);
                }
            }
        }
    }

    private double distance(Vector vec1, Vector vec2) {
        double distance = 0;
        for (int i = 0; i < vec1.length(); i++) {
            distance += Math.pow(vec1.get(i) - vec2.get(i), 2);
        }
        return distance;
    }

}
