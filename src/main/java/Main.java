import javafx.util.Pair;
import models.NeuralNetwork;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import utils.DataSetChart;

import java.util.LinkedList;
import java.util.List;

import static java.lang.Math.abs;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class Main {

    public static void main(String... args) {
        /*try {
            Matrix[] matrices;// = loadData("transformation.txt", "transformation.txt");
            //Matrix[] matrices = loadData("testData1.txt", null);
            NeuralNetwork nn = new NeuralNetwork(matrices[0], matrices[1], 3, true);
            nn.train(500);
            //Matrix test = new Basic2DMatrix(new double[][]{{10,10}});
            //Matrix result = nn.predict(test);
            //System.out.print(result.toString());
            //printDataSet(matrices);
            //printDataSetWithBoundry(matrices, nn);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }*/
        //new Transofrmation().performTransformation();
        new Approximation().performApproximation();
    }

    private static void printDataSet(Matrix[] matrices) {
        //TODO: numlabels jest hardcodowane
        DataSetChart chart = new DataSetChart(2);
        for (int i = 0; i < matrices[0].rows(); i++) {
            chart.addEntry((int) (matrices[1].get(i, 0)), matrices[0].get(i, 0), matrices[0].get(i, 1));
        }
        chart.generateChart("dataset.jpg");
    }

    private static void printDataSetWithBoundry(Matrix[] matrices, NeuralNetwork neuralNetwork) {
        //TODO: numlabels jest hardcodowane
        DataSetChart chart = new DataSetChart(3);
        for (int i = 0; i < matrices[0].rows(); i++) {
            chart.addEntry((int) (matrices[1].get(i, 0)), matrices[0].get(i, 0), matrices[0].get(i, 1));
        }
        List<Pair<Double, Double>> boundryPoints = boundryPoints(neuralNetwork);
        for (Pair<Double, Double> pair : boundryPoints) {
            chart.addEntry(2, pair.getKey(), pair.getValue());
        }
        chart.generateChart("datasetWithBoundry.jpg");
    }

    private static List<Pair<Double, Double>> boundryPoints(NeuralNetwork neuralNetwork) {
        List<Pair<Double, Double>> boundryPoints = new LinkedList<>();
        Matrix input = new Basic2DMatrix(10000, 2);
        double x = -5;
        double y = -5;
        for (int i = 0; i < 10000; i++) {
            input.set(i, 0, x);
            input.set(i, 1, y + 0.1 * (i % 100));
            if (i % 100 == 0) {
                x += 0.1;
            }
        }
        Matrix results = neuralNetwork.predict(input);
        for (int i = 0; i < 10000; i++) {
            if (abs(results.get(i, 0) - 0.5) < 0.1) {
                boundryPoints.add(new Pair<>(input.get(i, 0), input.get(i, 1)));
            }
        }
        return boundryPoints;
    }
}
