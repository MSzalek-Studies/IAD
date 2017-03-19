package activator;

import org.la4j.Matrix;

/**
 * Created by marcinus on 19.03.17.
 */
public class LinearActivator implements Activator {

    private final double MULTIPLIER;

    public LinearActivator() {
        this(1);
    }

    public LinearActivator(double MULTIPLIER) {
        this.MULTIPLIER = MULTIPLIER;
    }


    @Override
    public Matrix activate(Matrix matrix) {
        return matrix.multiply(MULTIPLIER);
    }
}
