package exercise2;

import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.vector.dense.BasicVector;
import utils.DataSetChart;

import java.util.HashMap;
import java.util.Random;

/**
 * Created by marcinus on 20.04.17.
 */
public class KMeans {

    int k;
    Matrix neurons;
    Matrix inputs;
    Vector ids;

    public KMeans(Matrix inputMatrix, int k) {
        this.k = k;
        initNeurons(inputMatrix, k);
        this.inputs = inputMatrix;
        this.ids = new BasicVector(inputMatrix.rows());
    }

    public void perform(int maxIter) {
        double diff;
        int it = 0;
        do {
            DataSetChart dataSetChart = new DataSetChart(2);
            dataSetChart.addEntries(0, inputs);
            dataSetChart.addEntries(1, neurons);
            dataSetChart.generateChart("ex2kmeans.jpg");
            System.out.println("generating ex2kmeans" + it);

            updateIds();
            removeDeadNeurons();
            diff = updateNeurons();

            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            it++;
        } while (diff > 0 && it < maxIter);
        DataSetChart dataSetChart = new DataSetChart(2);
        dataSetChart.addEntries(0, inputs);
        dataSetChart.addEntries(1, neurons);
        dataSetChart.generateChart("ex2kmeans.jpg");
        System.out.println("generating ex2kmeans" + it);
    }

    private void removeDeadNeurons() {
        HashMap<Integer, Integer> aliveNeuronIds = new HashMap<>();//Old index, new Index
        int newIndex = 0;
        //System.out.println(neurons.rows());
        for (int i = 0; i < ids.length(); i++) {
            Integer neuronId = aliveNeuronIds.get((int) ids.get(i));
            if (neuronId == null) {
                aliveNeuronIds.put((int) ids.get(i), newIndex++);
            }
        }
        //System.out.println(neurons.rows());
        if (k != aliveNeuronIds.size()) {
            k = aliveNeuronIds.size();
            Basic2DMatrix aliveNeurons = new Basic2DMatrix(k, 2);
            for (Integer i : aliveNeuronIds.keySet()) {
                //System.out.println("klucz"+i);
                aliveNeurons.set(aliveNeuronIds.get(i), 0, neurons.get(i, 0));
                aliveNeurons.set(aliveNeuronIds.get(i), 1, neurons.get(i, 1));
            }
            for (int i = 0; i < ids.length(); i++) {
                ids.set(i, aliveNeuronIds.get((int) ids.get(i)));
            }
            neurons = aliveNeurons;
        }

    }

    private double updateNeurons() {
        double totalDiff = 0;
        for (int i = 0; i < k; i++) {
            double meanX = 0;
            double meanY = 0;
            int counter = 0;
            for (int j = 0; j < ids.length(); j++) {
                if (ids.get(j) == i) {
                    meanX += inputs.get(j, 0);
                    meanY += inputs.get(j, 1);
                    counter++;
                }
            }
            if (counter > 0) {
                meanX /= counter;
                meanY /= counter;

                if (totalDiff < 10) {
                    totalDiff += Math.abs(neurons.get(i, 0) - meanX);
                    totalDiff += Math.abs(neurons.get(i, 1) - meanY);
                }
                neurons.set(i, 0, meanX);
                neurons.set(i, 1, meanY);
            }
        }
        return totalDiff;
    }

    private void updateIds() {
        for (int i = 0; i < inputs.rows(); i++) {
            double minDistance = -1;
            for (int j = 0; j < k; j++) {
                double distance = distance(neurons.get(j, 0), neurons.get(j, 1), inputs.get(i, 0), inputs.get(i, 1));
                if (minDistance == -1 || distance < minDistance) {
                    minDistance = distance;
                    ids.set(i, j);
                }
            }
        }
    }

    private double distance(double x1, double y1, double x2, double y2) {
        return Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2);
    }

    private void initNeurons(Matrix inputMatrix, int k) {
        neurons = new Basic2DMatrix(k, 2);
        double minXArg = inputMatrix.getColumn(0).min();
        double maxXArg = inputMatrix.getColumn(0).max();
        double minYArg = inputMatrix.getColumn(1).min();
        double maxYArg = inputMatrix.getColumn(1).max();
        for (int i = 0; i < k; i++) {
            double x = new Random().nextDouble() * (maxXArg - minXArg) + minXArg;
            double y = new Random().nextDouble() * (maxYArg - minYArg) + minYArg;
            neurons.set(i, 0, x);
            neurons.set(i, 1, y);
        }
    }
}
