package centroids;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.TreeSet;
import org.apache.commons.math3.util.MathArrays;
import contrast.Callback;
import contrast.DistanceObject;
import streamDataStructures.Selection;
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
	 * A ranking of the indexes of the {@link Centroid} according to their kNN
	 * distances.
	 */
	private double[] kNNRank;
	/**
	 * The {@link Callback} to alarm on changes found out checking the kNN
	 * distribution.
	 */
	private Callback callback;
	/**
	 * The total weight for a kNN search.
	 */
	private double k;
	/**
	 * The allowed change in the kNNRank without alarming the callback.
	 */
	private double allowedChange;
	/**
	 * Keeps track of the current time.
	 */
	private int time = 0;
	/**
	 * Keeps track of how many {@link Instance}s arrived since the last check of
	 * the kNN rank.
	 */
	private double count = 0;
	/**
	 * Determines after how many new {@link Instance}s the kNNRank is checked.
	 */
	private double checkCount;
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
	 * @param k
	 *            The total weight for the kNNN calculation.
	 * @param checkCount
	 *            The amount of new {@link Instance}s after which the kNN rank
	 *            is checkded.
	 * @param allowedChange
	 *            The allowed change in the kNN rank before an alarm to the
	 *            {@link Callback} is signalled.
	 * @param weigthThreshold
	 *            The threshold which determines when a {@link Centroid} is
	 *            removed.
	 */
	public AdaptingCentroids(Callback callback, int numberOfDimensions, double fadingLambda, double radius, double k,
			int checkCount, double allowedChange, double weigthThreshold) {
		this.centroids = new ArrayList<Centroid>();
		this.callback = callback;
		this.numberOfDimensions = numberOfDimensions;
		this.fadingFactor = Math.pow(2, -fadingLambda);
		this.radius = radius;
		this.k = k;
		this.checkCount = checkCount;
		this.allowedChange = allowedChange;
		this.weightThreshold = weigthThreshold;
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
			nearest = new Centroid(vector, fadingFactor, time);
			centroids.add(nearest);
		} else {
			adaptCentroid(nearest, vector);
		}
		// Increment count and weight of centroid. It was already faded before.
		nearest.increment();

		time++;
		updated = false;

		count++;
		if (count >= checkCount) {
			densityCheck();
			count = 0;
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
			distance = euclideanDistance(c.getVector(), vector);
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
		count = 0;
	}

	@Override
	public int getNumberOfInstances() {
		return centroids.size();
	}

	@Override
	public double[] getProjectedData(int referenceDimension) {
		if (!updated) {
			updateWeights();
		}
		int totalWeight = 0;
		for (Centroid c : centroids) {
			totalWeight += Math.round(c.getWeight());
		}

		double[] projectedData = new double[totalWeight];
		int index = 0;
		for (Centroid c : centroids) {
			for (int j = 0; j < Math.round(c.getWeight()); j++) {
				projectedData[index] = c.getVector()[referenceDimension];
				index++;
			}
		}
		return projectedData;
	}

	@Override
	public double[] getSlicedData(int[] shuffledDimensions, double selectionAlpha) {
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

		// Get the selected data from the last dimension and apply weights
		weights = getSelectedWeights(selectedIndexes);
		int totalWeight = 0;
		for (int i = 0; i < weights.length; i++) {
			weights[i] = Math.round(weights[i]);
			totalWeight += weights[i];
		}
		double[] slicedData = new double[totalWeight];
		int index = 0;
		Centroid c;
		int referenceDimension = shuffledDimensions[shuffledDimensions.length - 1];
		for (int i = 0; i < selectedIndexes.size(); i++) {
			c = centroids.get(selectedIndexes.getIndex(i));
			for (int j = 0; j < weights[i]; j++) {
				slicedData[index] = c.getVector()[referenceDimension];
				index++;
			}
		}
		return slicedData;
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
	public void densityCheck() {
		if (!updated) {
			updateWeights();
		}
		int numberOfCentroids = centroids.size();
		// Calculate kNN-distances and sort them.
		double[] kNNDistances = new double[numberOfCentroids];
		double[] indexes = new double[numberOfCentroids];

		for (int i = 0; i < numberOfCentroids; i++) {
			kNNDistances[i] = calculateKNNDistance(i, k);
			indexes[i] = i;
		}

		MathArrays.sortInPlace(kNNDistances, indexes);

		// Calculate the change in the kNN rank
		int change = 0;
		if (kNNRank != null) {
			for (int i = 0; i < numberOfCentroids; i++) {
				for (int j = 0; j < numberOfCentroids; j++) {
					if (kNNRank[i] == indexes[j]) {
						change += Math.abs(i - j);
					}
				}
			}
		}

		kNNRank = indexes;

		if (change > allowedChange * numberOfCentroids) {
			// Notify callback to check contrast
			callback.onAlarm();
		}

	}

	/**
	 * Calculates the kNN distance for the {@linkCentroid} by the given index. k
	 * does not indicates how many neighbouring {@linkCentroid}s have to be
	 * taken into account, but how big the accumulated weight of the
	 * neighbouring {@link Centroid}s has to be.
	 * 
	 * @param index
	 *            The index of the {@link Centroid} for which the kNN distance
	 *            should be calculated.
	 * @param k
	 *            The accumulated weight of the neighbours have to have.
	 * @return The kNN distance, where k is the accumulated weight that has to
	 *         be reached.
	 */
	private double calculateKNNDistance(int index, double k) {
		Centroid c = centroids.get(index);
		TreeSet<DistanceObject> distSet = new TreeSet<DistanceObject>(new Comparator<DistanceObject>() {
			@Override
			public int compare(DistanceObject o1, DistanceObject o2) {
				if (o1.getDistance() < o2.getDistance()) {
					return -1;
				} else if (o1.getDistance() > o2.getDistance()) {
					return 1;
				}
				return 0;
			}
		});

		for (int i = 0; i < centroids.size(); i++) {
			if (i != index) {
				distSet.add(new DistanceObject(euclideanDistance(c.getVector(), centroids.get(i).getVector()),
						centroids.get(i).getWeight()));
			}
		}

		double accumulatedWeight = 0;
		Iterator<DistanceObject> it = distSet.descendingIterator();
		double kNNDistance = 0;
		DistanceObject d;
		while (it.hasNext() && accumulatedWeight < k) {
			d = it.next();
			accumulatedWeight += d.getWeight();
			kNNDistance = d.getDistance();
		}

		return kNNDistance;
	}
}
