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
public class ErrorChart {

    XYSeriesCollection dataset = new XYSeriesCollection();
    private XYSeries series = new XYSeries("XYGraph");

    public void addEntry(double x, double y) {
        series.add(x, y);
    }

    public void addSeries(XYSeries series) {
        dataset.addSeries(series);
    }

    public void generateChart() {
        // Add the series to your data set
        dataset.addSeries(series);

        // Generate the graph
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Wykres spadku błędu", // Title
                "iteracja", // x-axis Label
                "błąd", // y-axis Label
                dataset, // Dataset
                PlotOrientation.VERTICAL, // Plot Orientation
                false, // Show Legend
                false, // Use tooltips
                false // Configure chart to generate URLs?
        );
        try {
            ChartUtilities.saveChartAsJPEG(new File("chart.jpg"), chart, 500, 300);
        } catch (IOException e) {
            System.err.println("Problem occurred creating chart.");
        }
    }
}
