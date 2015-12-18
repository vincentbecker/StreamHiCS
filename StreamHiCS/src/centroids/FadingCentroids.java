package centroids;

import java.util.ArrayList;

import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.tasks.TaskMonitor;
import weka.core.Instance;

/**
 * This class represents a {@link AdaptingCentroid} implementation which adapts
 * to the input data.
 * 
 * @author Vincent
 *
 */
public class FadingCentroids extends AbstractOptionHandler {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	/**
	 * The centroids.
	 */
	private ArrayList<Centroid> centroids;
	/**
	 * The number of dimensions of the full space.
	 */
	private int numberOfDimensions = -1;

	private double negLambda;
	/**
	 * For each incoming instance we search for the nearest
	 * {@link AdaptingCentroid} in which radius the instance falls.
	 */
	private double radius;
	/**
	 * Keeps track of the current time.
	 */
	private int time = 0;
	/**
	 * The threshold determining when a {@link AdaptingCentroid} is removed.
	 */
	private final double weightThreshold = 0.25;
	/**
	 * A flag showing if all weights are updated currently.
	 */
	private boolean updated = true;
	/**
	 * The learning rate for the adaptation of the {@link AdaptingCentroid}.
	 */
	private double learningRate;
	private boolean initialized = false;

	public IntOption horizonOption = new IntOption("horizon", 'h', "Horizon", 1000, 1, Integer.MAX_VALUE);

	/**
	 * The radius of a {@link AdaptingCentroid} determining if a
	 * new @link{Instance} could belong to it.
	 */
	public FloatOption radiusOption = new FloatOption("radius", 'r', "Radius", 1, 0, Double.MAX_VALUE);

	/**
	 * The learning rate.
	 */
	public FloatOption learningRateOption = new FloatOption("scale", 's', "Scale.", 1, 0, Double.MAX_VALUE);

	public Centroid[] getCentroids() {
		updateWeights();
		AdaptingCentroid[] cs = new AdaptingCentroid[centroids.size()];
		centroids.toArray(cs);
		return cs;
	}

	public void add(Instance instance) {
		if (!initialized) {
			this.numberOfDimensions = instance.numAttributes();
			initialized = true;
		} else {
			if (instance.numAttributes() != numberOfDimensions) {
				throw new IllegalArgumentException("Instance has wrong number of dimensions. ");
			}
		}
		// Create vector
		double[] vector = instance.toDoubleArray();

		// Find the closest centroid
		Centroid nearest = findNearestCentroid(vector);
		if (nearest == null) {
			nearest = new AdaptingCentroid(vector, negLambda, time, radius, learningRate);
			centroids.add(nearest);
		} else {
			if (!nearest.addPoint(vector, time)) {
				centroids.add(new AdaptingCentroid(vector, negLambda, time, radius, learningRate));
			}
		}
		time++;
		updated = false;
	}

	/**
	 * Finds the nearest centroid to the given vector. The vector has to fall in
	 * the radius of the {@link AdaptingCentroid}. Before searching fo the
	 * nearest {@link AdaptingCentroid} the weights of all
	 * {@link AdaptingCentroid}s are updated.
	 * 
	 * @param vector
	 *            The input vector.
	 * @return The nearest {@link AdaptingCentroid} to the given vector, in
	 *         which's radius the vector falls. Null, if such a vector does not
	 *         exist.
	 */
	private Centroid findNearestCentroid(double[] vector) {
		double distance;
		double minDistance = Double.MAX_VALUE;
		Centroid nearestCentroid = null;

		updateWeights();

		for (Centroid c : centroids) {
			distance = c.euclideanDistance(vector);
			if (distance < minDistance) {
				minDistance = distance;
				nearestCentroid = c;
			}
		}
		return nearestCentroid;
	}

	/**
	 * Updates the weights of all contained {@link AdaptingCentroid}s by
	 * multiplying the fading factor to the power of the difference between the
	 * current time and the last time the {@link AdaptingCentroid} was updated.
	 * If the weight falls below the weight threshold the
	 * {@link AdaptingCentroid} is removed.
	 */
	private void updateWeights() {
		if (!updated) {
			ArrayList<Centroid> removalList = new ArrayList<Centroid>();
			for (Centroid c : centroids) {
				// c.fade(time);
				if (c.getWeight(time) < weightThreshold) {
					removalList.add(c);
				}
			}
			for (Centroid c : removalList) {
				centroids.remove(c);
			}
			updated = true;
		}
	}

	public void clear() {
		centroids.clear();
		time = 0;
		initialized = false;
		numberOfDimensions = -1;
	}

	public int getNumberOfInstances() {
		updateWeights();
		return centroids.size();
	}

	@Override
	public void getDescription(StringBuilder arg0, int arg1) {
		// TODO Auto-generated method stub

	}

	@Override
	protected void prepareForUseImpl(TaskMonitor arg0, ObjectRepository arg1) {
		this.centroids = new ArrayList<Centroid>();
		// Calculating the fading scale log2(threshold) / horizon. Here the log2
		// is -2. Beware, relies on weight threshold being 0.25
		this.negLambda = -2.0 / horizonOption.getValue();
		this.radius = radiusOption.getValue();
		this.learningRate = learningRateOption.getValue();
		initialized = false;
		time = 0;
	}
}