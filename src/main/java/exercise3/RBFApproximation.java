package exercise3;

import org.jfree.data.xy.XYSeries;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic1DMatrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.functor.MatrixFunction;
import utils.DataSetChart;
import utils.ErrorChart;
import utils.FileUtils;
import utils.MatrixUtils;

import java.io.IOException;
import java.util.Random;

/**
 * Created by marcinus on 19.03.17.
 */
public class RBFApproximation {

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

    public void performApproximation(int iterations, int hiddenNeurons, double learningRate) throws IOException {
        Matrix[] matrices = new FileUtils().loadDataFromSingleFileSupervised("approximation_train_1.txt");
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

        initRandomCenters();
        initSigmas();

        ErrorChart errorChart = new ErrorChart();
        XYSeries series = new XYSeries("Seria");

        for (int it = 0; it < iterations; it++) {
            propagateForwards();
            outputErrors = outputs.subtract(expectedValues);
            outputWeightsErrors = rbfValues.transpose().multiply(outputErrors);
            Matrix difference = outputWeightsErrors.multiply(learningRate);
            outputWeights = outputWeights.subtract(difference);
            double cost = outputErrors.transform(new MatrixFunction() {
                @Override
                public double evaluate(int i, int j, double value) {
                    return (value * value) / (2 * m);
                }
            }).sum(); //TODO: dzielic przez 2?
            series.add(it, cost);
            System.out.println(cost);
        }
        errorChart.addSeries(series);
        errorChart.generateChart("nazwaaa.png");
        show2DDataAndApproximation();
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
            outputs = rbfValues.multiply(outputWeights);
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

    private void show2DDataAndApproximation() {
        DataSetChart chart = new DataSetChart(2);
        for (int i = 0; i < inputs.rows(); i++) {
            chart.addEntry(0, inputs.get(i, 0), expectedValues.get(i, 0));
        }
        double minArg = inputs.min();
        double maxArg = inputs.max();
        double step = (maxArg - minArg) / 100;
        Matrix testMatrix = new Basic1DMatrix(101, 1, new double[101]);
        int index = 0;
        for (double i = minArg; i < maxArg; i += step) {
            testMatrix.set(index, 0, i);
            index++;
        }
        predict(testMatrix);
        index = 0;
        for (double i = minArg; i < maxArg; i += step) {
            chart.addEntry(1, i, outputs.get(index, 0));
            index++;
        }
        chart.generateChart("approxData.jpg");
    }

}
