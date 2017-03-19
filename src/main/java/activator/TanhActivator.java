package activator;

import org.la4j.Matrix;

/**
 * Created by marcinus on 19.03.17.
 */
public class TanhActivator implements Activator {

    private final int VALUE_MULTIPLIER;

    public TanhActivator() {
        this(1);
    }

    public TanhActivator(int multiplier) {
        VALUE_MULTIPLIER = multiplier;
    }


    @Override
    public Matrix activate(Matrix matrix) {
        return matrix.transform((i, i1, v) -> Math.tanh(100 * v)).multiply(VALUE_MULTIPLIER);
    }
}
