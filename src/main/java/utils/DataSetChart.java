package utils;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.File;
import java.io.IOException;

/**
 * Created by marcinus on 16.03.17.
 */
public class DataSetChart {

    private XYSeries[] series;

    public DataSetChart(int numLabels) {
        series = new XYSeries[numLabels];
        for (int i = 0; i < numLabels; i++) {
            series[i] = new XYSeries("label" + i);
        }
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
        JFreeChart chart = ChartFactory.createScatterPlot(
                "Punkty", "X", "Y", dataset,
                PlotOrientation.VERTICAL, true, true, false);

        try {
            ChartUtilities.saveChartAsJPEG(new File(filename), chart, 500, 300);
        } catch (IOException e) {
            System.err.println("Problem occurred creating chart.");
        }
    }
}
