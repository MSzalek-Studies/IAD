package exercise2;

import org.jfree.data.xy.XYSeries;
import org.la4j.Matrix;
import utils.DataSetChart;
import utils.ErrorChart;

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

    /**
     * @param maxIter
     * @param learningRate
     */
    public void perform(int maxIter, double learningRate, double lambda) {
        double diff;
        int it = 0;
        try {
            org.apache.commons.io.FileUtils.deleteDirectory(new File("ngas"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        new File("ngas").mkdir();
        ErrorChart errorChart = new ErrorChart();
        XYSeries errorSeries = new XYSeries("Bledy");
        do {
            DataSetChart dataSetChart = new DataSetChart(2);
            points.forEach(p -> dataSetChart.addEntry(0, p.x, p.y));
            neurons.forEach(p -> dataSetChart.addEntry(1, p.x, p.y));
            dataSetChart.generateChart("ngas" + File.separator + "ex2ngas" + it + ".jpg");
            System.out.println("generating ngas" + it);

            setWinners();
            diff = moveWinners(learningRate, lambda);
            if (it > 5) {
                removeDeadNeurons();
            }
            errorSeries.add(it, calculateError());
            it++;
        } while (diff > 0 && it < maxIter);
        errorChart.addSeries(errorSeries);
        errorChart.generateChart("ngas" + File.separator + "errorchart.jpg");
    }

    private void removeDeadNeurons() {
        neurons.removeIf(n -> !points.stream().map(p -> p.neurons.get(0)).collect(Collectors.toList()).contains(n));
    }

    private double calculateError() {
        return points.stream().mapToDouble(p -> distance(p.x, p.y, p.neurons.get(0).x, p.neurons.get(0).y)).average().getAsDouble();
    }

    private double distance(double x1, double y1, double x2, double y2) {
        return Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2);
    }

    private double distance(Point p1, Point p2) {
        return distance(p1.x, p1.y, p2.x, p2.y);
    }

    private double moveWinners(double learningRate, double lambda) {
        double[] diff = {0};
        points.forEach(p -> {
            int[] it = {0};
            p.neurons.forEach(n -> {
                double neighbourFactor = Math.exp(-it[0] / lambda);
                double diffX = neighbourFactor * learningRate * (p.x - n.x);
                double diffY = neighbourFactor * learningRate * (p.y - n.y);
                n.x += diffX;
                n.y += diffY;
                diff[0] += diffX + diffY;
                it[0]++;
            });
        });
        return Math.abs(diff[0]);
    }

    private void setWinners() {
        points.forEach(p -> {
            p.neurons = new ArrayList<>(neurons);
            p.neurons.sort((n1, n2) -> (distance(p, n1) - distance(p, n2)) > 0 ? 1 : -1);
        });
    }

    class TrainPoint extends Point {
        List<Point> neurons = new ArrayList<>();

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
}

