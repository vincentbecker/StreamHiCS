package centroids;

import java.util.ArrayList;

import changechecker.ChangeChecker;
import contrast.Callback;
import contrast.DataBundle;
import contrast.Selection;
import weka.core.Instance;

/**
 * This class represents a {@link Centroid} implementation which adapts to the
 * input data.
 * 
 * @author Vincent
 *
 */
public class AdaptingCentroids extends CentroidsContainer {
	/**
	 * The centroids.
	 */
	private ArrayList<Centroid> centroids;
	/**
	 * The number of dimensions of the full space.
	 */
	private int numberOfDimensions;
	/**
	 * The fading factor the weight of every {@link Centroid} is multiplied with
	 * after each time step.
	 */
	private double fadingFactor;
	/**
	 * For each incoming instance we search for the nearest {@link Centroid} in
	 * which radius the instance falls.
	 */
	private double radius;
	/**
	 * The {@link Callback} to alarm on changes found out checking the kNN
	 * distribution.
	 */
	private Callback callback;
	/**
	 * Keeps track of the current time.
	 */
	private int time = 0;
	/**
	 * The threshold determining when a {@link Centroid} is removed.
	 */
	private double weightThreshold;
	/**
	 * A flag showing if all weights are updated currently.
	 */
	private boolean updated = true;
	/**
	 * The learning rate for the adaptation of the {@link Centroid}.
	 */
	private double learningRate = 0.1;
	/**
	 * The {@link ChangeChecker}
	 */
	private ChangeChecker changeChecker;

	/**
	 * Created an object of this class.
	 * 
	 * @param callback
	 *            The {@link Callback} to alarm on changes in the kNN rank.
	 * @param numberOfDimensions
	 *            The number of dimensions in the full space.
	 * @param fadingLambda
	 *            The fading lambda.
	 * @param radius
	 *            The radius of a {@link Centroid} determining if a
	 *            new @link{Instance} could belong to it.
	 * @param checkCount
	 *            The amount of new {@link Instance}s after which we check for
	 *            change.
	 * @param weigthThreshold
	 *            The threshold which determines when a {@link Centroid} is
	 *            removed.
	 * @param learningRate
	 *            The learning rate.
	 */
	public AdaptingCentroids(Callback callback, int numberOfDimensions, double fadingLambda, double radius,
			double weigthThreshold, double learningRate) {
		this.centroids = new ArrayList<Centroid>();
		this.callback = callback;
		this.numberOfDimensions = numberOfDimensions;
		this.fadingFactor = Math.pow(2, -fadingLambda);
		this.radius = radius;
		this.weightThreshold = weigthThreshold;
		this.learningRate = learningRate;
	}

	@Override
	public Centroid[] getCentroids() {
		Centroid[] cs = new Centroid[centroids.size()];
		centroids.toArray(cs);
		return cs;
	}

	@Override
	public void add(Instance instance) {
		// Create vector
		double[] vector = new double[numberOfDimensions];
		for (int i = 0; i < numberOfDimensions; i++) {
			vector[i] = instance.value(i);
		}

		// Find the closest centroid
		Centroid nearest = findNearestCentroid(vector);
		if (nearest == null) {
			nearest = new Centroid(time, vector, fadingFactor, time);
			centroids.add(nearest);
		} else {
			adaptCentroid(nearest, vector);
		}
		// Increment count and weight of centroid. It was already faded before.
		nearest.increment();

		time++;
		updated = false;

		if (changeChecker.poll()) {
			changeCheck();
		}
	}

	/**
	 * Finds the nearest centroid to the given vector. The vector has to fall in
	 * the radius of the {@link Centroid}. Before searching fo the nearest
	 * {@link Centroid} the weights of all {@link Centroid}s are updated.
	 * 
	 * @param vector
	 *            The input vector.
	 * @return The nearest {@link Centroid} to the given vector, in which's
	 *         radius the vector falls. Null, if such a vector does not exist.
	 */
	private Centroid findNearestCentroid(double[] vector) {
		double distance;
		double minDistance = Double.MAX_VALUE;
		Centroid nearestCentroid = null;

		updateWeights();

		for (Centroid c : centroids) {
			distance = c.euclideanDistance(vector);
			if (distance < radius && distance < minDistance) {
				minDistance = distance;
				nearestCentroid = c;
			}
		}
		return nearestCentroid;
	}

	/**
	 * Moves a given {@link Centroid} towards a given vector.
	 * 
	 * @param c
	 *            The {@link Centroid} that should be adapted.
	 * @param vector
	 *            The input vector.
	 */
	private void adaptCentroid(Centroid c, double[] vector) {
		double[] centroidVector = c.getVector();
		double adaptationRate = h(c, vector);
		for (int i = 0; i < numberOfDimensions; i++) {
			centroidVector[i] += adaptationRate * (vector[i] - centroidVector[i]);
		}
	}

	/**
	 * Calculates a factor how strongly a {@link Centroid} should be adapted to
	 * a vector.
	 * 
	 * @param c
	 *            The {@link Centroid}.
	 * @param vector
	 *            The input vector.
	 * @return The factor how strongly the given {@link Centroid} should be
	 *         adapted to the input vector.
	 */
	private double h(Centroid c, double[] vector) {
		return learningRate;
	}

	/**
	 * Updates the weights of all contained {@link Centroid}s by multiplying the
	 * fading factor to the power of the difference between the current time and
	 * the last time the {@link Centroid} was updated. If the weight falls below
	 * the weight threshold the {@link Centroid} is removed.
	 */
	private void updateWeights() {
		if (!updated) {
			ArrayList<Centroid> removalList = new ArrayList<Centroid>();
			for (Centroid c : centroids) {
				c.fade(time);
				if (c.getWeight() < weightThreshold) {
					removalList.add(c);
				}
			}
			for (Centroid c : removalList) {
				centroids.remove(c);
			}
			updated = true;
		}
	}

	@Override
	public void clear() {
		centroids.clear();
		time = 0;
	}

	@Override
	public int getNumberOfInstances() {
		return centroids.size();
	}

	@Override
	public DataBundle getProjectedData(int referenceDimension) {
		if (!updated) {
			updateWeights();
		}

		int l = centroids.size();

		double[] data = new double[l];
		double[] weights = new double[l];
		Centroid c;
		for (int i = 0; i < l; i++) {
			c = centroids.get(i);
			data[i] = c.getVector()[referenceDimension];
			weights[i] = c.getWeight();
		}
		return new DataBundle(data, weights);
	}

	@Override
	public DataBundle getSlicedData(int[] shuffledDimensions, double selectionAlpha) {
		if (!updated) {
			updateWeights();
		}

		double[] dimData;
		double[] weights;
		Selection selectedIndexes = new Selection(centroids.size(), selectionAlpha);
		// Fill the list with all the indexes
		selectedIndexes.fillRange();

		for (int i = 0; i < shuffledDimensions.length - 1; i++) {
			// Get all the data for the specific dimension that is selected
			dimData = getSelectedData(shuffledDimensions[i], selectedIndexes);
			weights = getSelectedWeights(selectedIndexes);
			// Reduce the number of indexes according to a new selection in
			// the current dimension
			selectedIndexes.selectWithWeights(dimData, weights);
		}

		// Get the selected data from the last dimension and apply weights
		dimData = getSelectedData(shuffledDimensions[shuffledDimensions.length - 1], selectedIndexes);
		weights = getSelectedWeights(selectedIndexes);

		return new DataBundle(dimData, weights);
	}

	// TODO: Remove
	public Selection getSliceIndexes(int[] shuffledDimensions, double selectionAlpha) {
		if (!updated) {
			updateWeights();
		}

		double[] dimData;
		double weights[];
		Selection selectedIndexes = new Selection(centroids.size(), selectionAlpha);
		// Fill the list with all the indexes
		selectedIndexes.fillRange();

		for (int i = 0; i < shuffledDimensions.length - 1; i++) {
			// Get all the data for the specific dimension that is selected
			dimData = getSelectedData(shuffledDimensions[i], selectedIndexes);
			weights = getSelectedWeights(selectedIndexes);
			// Reduce the number of indexes according to a new selection in
			// the current dimension
			selectedIndexes.selectWithWeights(dimData, weights);
		}
		return selectedIndexes;
	}

	/**
	 * Gets the data contained in the selected {@link Centroid}s according to
	 * the given indexes and projected to the given dimension.
	 * 
	 * @param dimension
	 *            The dimension the data is projected to.
	 * @param selectedIndexes
	 *            The selection indexes.
	 * @return A double[] containing the projected data from the selected
	 *         {@link Centroid}s.
	 */
	public double[] getSelectedData(int dimension, Selection selectedIndexes) {
		double[] data = new double[selectedIndexes.size()];
		for (int i = 0; i < selectedIndexes.size(); i++) {
			data[i] = centroids.get(selectedIndexes.getIndex(i)).getVector()[dimension];
		}
		return data;
	}

	/**
	 * Gets the weights of the selected {@link Centroid}s.
	 * 
	 * @param selectedIndexes
	 *            The selection indexes.
	 * @return A double[] containing the weights of the selected
	 *         {@link Centroid}s.
	 */
	private double[] getSelectedWeights(Selection selectedIndexes) {
		double[] weights = new double[selectedIndexes.size()];
		for (int i = 0; i < selectedIndexes.size(); i++) {
			weights[i] = centroids.get(selectedIndexes.getIndex(i)).getWeight();
		}
		return weights;
	}

	@Override
	public void changeCheck() {
		if (!updated) {
			updateWeights();
		}
		// Check for change and inform the callback in case.
		if (changeChecker.poll()) {
			callback.onAlarm();
		}
	}
}
