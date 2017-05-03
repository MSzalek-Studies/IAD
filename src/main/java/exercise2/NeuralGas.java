package exercise2;

import org.la4j.Matrix;
import org.la4j.Vector;
import utils.DataSetChart;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by marcinus on 26.04.17.
 */
public class NeuralGas {

    int k;
    List<Point> neurons = new ArrayList<>();
    List<TrainPoint> points = new ArrayList<>();
    Vector ids;

    class TrainPoint extends Point {
        Point neuron;

        public TrainPoint(double x, double y) {
            super(x, y);
        }
    }

    class Point {
        double x;
        double y;

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }

        double distance(Point point) {
            return Math.pow(point.x - x, 2) + Math.pow(point.y - y, 2);
        }
    }

    public NeuralGas(Matrix inputMatrix, int k) {
        this.k = k;
        Matrix neurons = new Utils().initNeurons(inputMatrix, k);
        for (int i = 0; i < neurons.rows(); i++) {
            this.neurons.add(new Point(neurons.get(i, 0), neurons.get(i, 1)));
        }
        for (int i = 0; i < inputMatrix.rows(); i++) {
            points.add(new TrainPoint(inputMatrix.get(i, 0), inputMatrix.get(i, 1)));
        }
    }

    public void perform(int maxIter, double learningRate) {
        double diff;
        int it = 0;
        try {
            org.apache.commons.io.FileUtils.deleteDirectory(new File("kohonen"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        new File("kohonen").mkdir();
        do {
            DataSetChart dataSetChart = new DataSetChart(2);
            points.forEach(p -> dataSetChart.addEntry(0, p.x, p.y));
            neurons.forEach(p -> dataSetChart.addEntry(1, p.x, p.y));
            dataSetChart.generateChart("kohonen" + File.separator + "ex2kohonen" + it + ".jpg");
            System.out.println("generating ex2kohonen" + it);

            setWinners();
            diff = moveWinners(learningRate);
            if (it > 5) {
                removeDeadNeurons();
            }
            //diff = updateNeurons();
            it++;
        } while (diff > 0 && it < maxIter);
    }

    private void removeDeadNeurons() {
        neurons.removeIf(n -> !points.stream().map(p -> p.neuron).collect(Collectors.toList()).contains(n));
    }

    private double moveWinners(double learningRate) {
        double[] diff = {0};
        points.forEach(p -> {
            neurons.forEach(n -> {
                double diffX = gaussian(p.neuron, n, 0.01) * learningRate * (p.x - n.x);
                double diffY = gaussian(p.neuron, n, 0.01) * learningRate * (p.y - n.y);
                n.x += diffX;
                n.y += diffY;
                diff[0] += diffX + diffY;
            });

        });
        return Math.abs(diff[0]);
    }

    private void setWinners() {
        points.forEach(p ->
                p.neuron = neurons.stream().min((n1, n2) -> (int) (p.distance(n1) - p.distance(n2))).get());
    }

    // return pdf(x) = standard Gaussian pdf
    public double gaussian(Point center, Point point, double sigmaSquared) {
        double value = Math.exp(-((Math.pow(center.x - point.x, 2) / (2 * sigmaSquared)) + ((Math.pow(center.y - point.y, 2) / (2 * sigmaSquared)))));
        return value;
    }

}

