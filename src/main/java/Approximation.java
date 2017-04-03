import activator.LinearActivator;
import models.NeuralNetwork;
import org.jfree.data.xy.XYSeries;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic1DMatrix;
import utils.DataSetChart;
import utils.ErrorChart;
import utils.FileUtils;

import java.io.FileNotFoundException;

/**
 * Created by marcinus on 19.03.17.
 */
public class Approximation {

    public void performApproximation() {
        try {
            Matrix[] matrices = new FileUtils().loadDataFromSingleFile("approximation_train_1.txt");
            NeuralNetwork nn = new NeuralNetwork(matrices[0], matrices[1], true,
                    new int[]{5}, new LinearActivator());
            ErrorChart errorChart = new ErrorChart();
            XYSeries errorSeries = nn.train(5000, 0.01);
            errorChart.addSeries(errorSeries);
            errorChart.generateChart();
            show2DDataAndApproximation(matrices, nn);

            test(nn);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private void test(NeuralNetwork nn) throws FileNotFoundException {
        Matrix[] matrices = new FileUtils().loadDataFromSingleFile("approximation_test.txt");
        System.out.print("TEST: " + nn.test(matrices[0], matrices[1]));
    }

    private void show2DDataAndApproximation(Matrix[] matrices, NeuralNetwork nn) {
        DataSetChart chart = new DataSetChart(2);
        for (int i = 0; i < matrices[0].rows(); i++) {
            chart.addEntry(0, matrices[0].get(i, 0), matrices[1].get(i, 0));
        }
        double minArg = matrices[0].min();
        double maxArg = matrices[0].max();
        double step = (maxArg - minArg) / 100;
        for (double i = minArg; i < maxArg; i += step) {
            Matrix inputMatrix = new Basic1DMatrix(1, 1, new double[]{i});
            chart.addEntry(1, i, nn.predict(inputMatrix).get(0, 0));
        }

        chart.generateChart("approxData.jpg");
    }

    private void show2DData(Matrix[] matrices) {
        DataSetChart chart = new DataSetChart(1);
        for (int i = 0; i < matrices[0].rows(); i++) {
            chart.addEntry(0, matrices[0].get(i, 0), matrices[1].get(i, 0));
        }
        chart.generateChart("approxData.jpg");
    }
}
