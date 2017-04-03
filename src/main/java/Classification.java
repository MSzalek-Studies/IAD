import models.NeuralNetwork;
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
            Matrix[] matrices = new FileUtils().loadDataFromSingleFile("classification_train.txt");
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
        train(inputMatrix, outputMatrix, "train_0.png", 0);
        train(inputMatrix, outputMatrix, "train_1.png", 1);
        train(inputMatrix, outputMatrix, "train_2.png", 2);
        train(inputMatrix, outputMatrix, "train_3.png", 3);
    }

    private void trainTwoFeatures(Matrix inputMatrix, Matrix outputMatrix) throws FileNotFoundException {
        train(inputMatrix, outputMatrix, "train_01.png", 0, 1);
        train(inputMatrix, outputMatrix, "train_02.png", 0, 2);
        train(inputMatrix, outputMatrix, "train_03.png", 0, 3);
        train(inputMatrix, outputMatrix, "train_12.png", 1, 2);
        train(inputMatrix, outputMatrix, "train_13.png", 1, 3);
        train(inputMatrix, outputMatrix, "train_23.png", 2, 3);
    }

    private void trainThreeFeatures(Matrix inputMatrix, Matrix outputMatrix) throws FileNotFoundException {
        train(inputMatrix, outputMatrix, "train_012.png", 0, 1, 2);
        train(inputMatrix, outputMatrix, "train_013.png", 0, 1, 3);
        train(inputMatrix, outputMatrix, "train_023.png", 0, 2, 3);
        train(inputMatrix, outputMatrix, "train_123.png", 1, 2, 3);
    }

    private void trainFourFeatures(Matrix inputMatrix, Matrix outputMatrix) throws FileNotFoundException {
        train(inputMatrix, outputMatrix, "train_0123.png", 0, 1, 2, 3);
    }

    private void train(Matrix inputMatrix, Matrix outputMatrix, String outputName, int... indexes) throws FileNotFoundException {
        inputMatrix = chooseParameters(inputMatrix, indexes);
        NeuralNetwork nn = new NeuralNetwork(inputMatrix.columns(), outputMatrix.columns(), true,
                new int[]{3}, 0.003, 0.9);
        ErrorChart errorChart = new ErrorChart();
        XYSeries errorSeries = nn.train(inputMatrix, outputMatrix, 500, 0.01);
        errorChart.addSeries(errorSeries);
        errorChart.generateChart(outputName);

        Matrix[] testMatrices = new FileUtils().loadDataFromSingleFile("classification_test.txt");
        System.out.println(outputName + " TEST: " + test(nn, chooseParameters(testMatrices[0], indexes), testMatrices[1]));
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
