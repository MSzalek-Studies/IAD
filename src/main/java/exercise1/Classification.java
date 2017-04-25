package exercise1;

import exercise1.models.NeuralNetwork;
import org.jfree.data.xy.XYSeries;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import utils.ErrorChart;
import utils.FileUtils;

import java.io.FileNotFoundException;

/**
 * Created by marcinus on 19.03.17.
 */
public class Classification {

    public void performClassification() {
        try {
            Matrix[] matrices = new FileUtils().loadDataFromSingleFileSupervised("classification_train.txt");
            matrices[1] = unrollClassificationMatrix(matrices[1]);

            trainOneFeature(matrices[0], matrices[1]);
            trainTwoFeatures(matrices[0], matrices[1]);
            trainThreeFeatures(matrices[0], matrices[1]);
            trainFourFeatures(matrices[0], matrices[1]);

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private void trainOneFeature(Matrix inputMatrix, Matrix outputMatrix) throws FileNotFoundException {
        ErrorChart errorChart = new ErrorChart();
        errorChart.addSeries(train(inputMatrix, outputMatrix, "Cecha 1", 0));
        errorChart.addSeries(train(inputMatrix, outputMatrix, "Cecha 2", 1));
        errorChart.addSeries(train(inputMatrix, outputMatrix, "Cecha 3", 2));
        errorChart.addSeries(train(inputMatrix, outputMatrix, "Cecha 4", 3));
        errorChart.generateChart("JednaCecha.png");
    }

    private void trainTwoFeatures(Matrix inputMatrix, Matrix outputMatrix) throws FileNotFoundException {
        ErrorChart errorChart = new ErrorChart();
        errorChart.addSeries(train(inputMatrix, outputMatrix, "Cechy 1 i 2", 0, 1));
        errorChart.addSeries(train(inputMatrix, outputMatrix, "Cechy 1 i 3", 0, 2));
        errorChart.addSeries(train(inputMatrix, outputMatrix, "Cechy 1 i 4", 0, 3));
        errorChart.addSeries(train(inputMatrix, outputMatrix, "Cechy 2 i 3", 1, 2));
        errorChart.addSeries(train(inputMatrix, outputMatrix, "Cechy 2 i 4", 1, 3));
        errorChart.addSeries(train(inputMatrix, outputMatrix, "Cechy 3 i 4", 2, 3));
        errorChart.generateChart("DwieCechy.png");
    }

    private void trainThreeFeatures(Matrix inputMatrix, Matrix outputMatrix) throws FileNotFoundException {
        ErrorChart errorChart = new ErrorChart();
        errorChart.addSeries(train(inputMatrix, outputMatrix, "Cechy 1,2,3", 0, 1, 2));
        errorChart.addSeries(train(inputMatrix, outputMatrix, "Cechy 1,2,4", 0, 1, 3));
        errorChart.addSeries(train(inputMatrix, outputMatrix, "Cechy 1,3,4", 0, 2, 3));
        errorChart.addSeries(train(inputMatrix, outputMatrix, "Cechy 2,3,4", 1, 2, 3));
        errorChart.generateChart("TrzyCechy.png");
    }

    private void trainFourFeatures(Matrix inputMatrix, Matrix outputMatrix) throws FileNotFoundException {
        ErrorChart errorChart = new ErrorChart();
        errorChart.addSeries(train(inputMatrix, outputMatrix, "Cztery Cechy", 0, 1, 2, 3));
        errorChart.generateChart("CzteryCechy.png");
    }

    private XYSeries train(Matrix inputMatrix, Matrix outputMatrix, String outputName, int... indexes) throws FileNotFoundException {
        inputMatrix = chooseParameters(inputMatrix, indexes);
        NeuralNetwork nn = new NeuralNetwork(inputMatrix.columns(), outputMatrix.columns(), false,
                new int[]{20}, 0.003, 0.9);
        //ErrorChart errorChart = new ErrorChart();
        XYSeries errorSeries = nn.train(inputMatrix, outputMatrix, 500, 0.01);
        errorSeries.setKey(outputName);
        //errorChart.addSeries(errorSeries);
        //errorChart.generateChart(outputName);

        Matrix[] testMatrices = new FileUtils().loadDataFromSingleFileSupervised("classification_test.txt");
        System.out.println(outputName + " TEST: " + test(nn, chooseParameters(testMatrices[0], indexes), testMatrices[1]));
        return errorSeries;
    }


    private Matrix chooseParameters(Matrix matrix, int... indexes) {
        Matrix childMatrix = new Basic2DMatrix(matrix.rows(), indexes.length);
        for (int i = 0; i < indexes.length; i++) {
            childMatrix.setColumn(i, matrix.getColumn(indexes[i]));
        }
        return childMatrix;
    }

    private double test(NeuralNetwork nn, Matrix input, Matrix expectedResults) {
        Matrix predictions = nn.predict(input);
        int goodClassifications = 0;
        for (int i = 0; i < input.rows(); i++) {
            int expectedResult = (int) expectedResults.get(i, 0);
            int expectedIndex = expectedResult - 1;
            if (predictions.get(i, expectedIndex) == predictions.maxInRow(i)) {
                goodClassifications++;
            } else {
                /*System.out.println(input.getRow(i));
                System.out.println(expectedResult);
                System.out.println(predictions.getRow(i) + "\n");*/
            }
        }
        return (double) goodClassifications / (double) input.rows();
    }

    private Matrix unrollClassificationMatrix(Matrix expectedResults) {
        int numClasses = (int) expectedResults.max();
        Matrix matrix = new Basic2DMatrix(expectedResults.rows(), numClasses);
        for (int i = 0; i < matrix.rows(); i++) {
            int expectedResult = (int) expectedResults.get(i, 0);
            matrix.set(i, expectedResult - 1, 1);
        }
        return matrix;
    }
}
