package exercise3;

import java.io.IOException;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class Ex3 {

    public static void main(String... args) throws IOException {
        //  new exercise1.Transofrmation().performTransformation();
        new RBFApproximation().performApproximation(1000, 5, 0.001);
        // new RBFClassification().performClassification(100, 5, 0.01);
    }

}
