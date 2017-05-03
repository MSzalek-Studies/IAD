package exercise2;

import org.la4j.Matrix;
import utils.DataSetChart;
import utils.FileUtils;

import java.io.FileNotFoundException;

/**
 * Created by marcinus on 20.04.17.
 */
public class Ex2 {
    public static void main(String[] args) {
        try {
            Matrix input = new FileUtils().loadDataFromSingleFileUnsupervised("attract.txt");
            DataSetChart dataSetChart = new DataSetChart(2);
            dataSetChart.addEntries(0, input);
//            KMeans kMeans = new KMeans(input, 500);
//            kMeans.perform(100);
            //dataSetChart.addEntries(1, kMeans.neurons);
            //dataSetChart.generateChart("ex2data.jpg");
            Kohonen kohonen = new Kohonen(input, 500);
            kohonen.perform(100, 0.1);

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }


}
