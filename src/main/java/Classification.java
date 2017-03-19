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
            NeuralNetwork nn = new NeuralNetwork(matrices[0], matrices[1], true,
                    new int[]{20});
            ErrorChart errorChart = new ErrorChart();
            XYSeries errorSeries = nn.train(500, 0.01);
            errorChart.addSeries(errorSeries);
            errorChart.generateChart();
            //show2DDataAndApproximation(matrices, nn);
            Matrix[] testMatrices = new FileUtils().loadDataFromSingleFile("classification_test.txt");
            System.out.print("TEST: " + test(nn, testMatrices[0], testMatrices[1]));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
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
                System.out.println(input.getRow(i));
                System.out.println(expectedResult);
                System.out.println(predictions.getRow(i) + "\n");
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
