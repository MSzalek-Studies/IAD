package utils;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.la4j.Matrix;

import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.io.File;
import java.io.IOException;

/**
 * Created by marcinus on 16.03.17.
 */
public class DataSetChart {

    private XYSeries[] series;
    public JFreeChart chart;

    public DataSetChart(int numLabels) {
        series = new XYSeries[numLabels];
        series[0] = new XYSeries("Punkty wejściowe");
        series[1] = new XYSeries("Neurony");
    }

    public void addEntries(int label, Matrix matrix) {
        for (int i = 0; i < matrix.rows(); i++) {
            series[label].add(matrix.get(i, 0), matrix.get(i, 1));
        }
    }

    public void setEntries(Matrix matrix) {
        series[1].clear();
        addEntries(1, matrix);
    }

    public void addEntry(int label, double x, double y) {
        series[label].add(x, y);
    }

    public void generateChart(String filename) {
        XYSeriesCollection dataset = new XYSeriesCollection(); // Add the series to your data set
        for (int i = 0; i < series.length; i++) {
            dataset.addSeries(series[i]);
        }


        // Generate the graph
        chart = ChartFactory.createScatterPlot(
                "Punkty", "X", "Y", dataset,
                PlotOrientation.VERTICAL, true, true, false);

        XYPlot xyPlot = chart.getXYPlot();
        XYItemRenderer renderer = xyPlot.getRenderer();
        double inputSize = 5;
        double inputDelta = inputSize / 2.0;
        Shape shape1 = new Ellipse2D.Double(-inputDelta, -inputDelta, inputSize, inputSize);
        renderer.setSeriesShape(0, shape1);
        renderer.setSeriesPaint(0, Color.blue);
        double neuronSize = 8;
        double neuronDelta = neuronSize / 2.0;
        Shape shape2 = new Ellipse2D.Double(-neuronDelta, -neuronDelta, neuronSize, neuronSize);
        renderer.setSeriesShape(1, shape2);
        renderer.setSeriesPaint(1, Color.red);

        try {
            ChartUtilities.saveChartAsJPEG(new File(filename), chart, 1000, 800);
        } catch (IOException e) {
            System.err.println("Problem occurred creating chart.");
        }
    }
}
