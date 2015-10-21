package centroids;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.TreeSet;

import org.apache.commons.math3.util.MathArrays;

import contrast.Callback;
import contrast.DistanceObject;

public class DensityChecker extends ChangeChecker {

	/**
	 * A ranking of the {@link Centroid}s according to their kNN distances.
	 */
	private Centroid[] kNNRank;
	/**
	 * The total weight for a kNN search.
	 */
	private double k;
	/**
	 * The allowed change in the kNNRank without alarming the callback.
	 */
	private double allowedChange;

	/**
	 * Creates an object of the class.
	 * 
	 * @param k
	 *            The total weight for the kNN calculation.
	 * @param allowedChange
	 *            The allowed change in the kNN rank before an alarm to the
	 *            {@link Callback} is signalled.
	 */
	public DensityChecker(double k, double allowedChange) {
		this.k = k;
		this.allowedChange = allowedChange;
	}

	@Override
	public boolean checkForChange(ArrayList<Centroid> centroids) {
		int numberOfCentroids = centroids.size();
		// Calculate kNN-distances and sort them.
		double[] kNNDistances = new double[numberOfCentroids];
		double[] indexes = new double[numberOfCentroids];
		Centroid[] newKNNRank = new Centroid[numberOfCentroids];
		for (int i = 0; i < numberOfCentroids; i++) {
			kNNDistances[i] = calculateKNNDistance(centroids, i, k);
			indexes[i] = i;
		}

		MathArrays.sortInPlace(kNNDistances, indexes);
		for (int i = 0; i < numberOfCentroids; i++) {
			newKNNRank[i] = centroids.get((int) indexes[i]);
		}

		// Calculate the change in the kNN rank
		int change = 0;
		boolean found = false;
		ArrayList<Integer> checked = new ArrayList<Integer>();
		if (kNNRank != null) {
			for (int i = 0; i < kNNRank.length; i++) {
				found = false;
				for (int j = 0; j < numberOfCentroids && !found; j++) {
					if (kNNRank[i].getId() == newKNNRank[j].getId()) {
						change += Math.abs(i - j);
						found = true;
						checked.add(j);
					}
				}
				if (!found) {
					// Means the centroid which previously was at index i was
					// removed.
					change += kNNRank.length - i;
				}
			}
			// Checking on all centroids which are new
			for (int i = 0; i < numberOfCentroids; i++) {
				if (!checked.contains(i)) {
					change += numberOfCentroids - i;
				}
			}

		} else {
			kNNRank = newKNNRank;
			return true;
		}
		kNNRank = newKNNRank;

		// TODO: Remove
		System.out.println("Change: " + change);
		//for (Centroid c : kNNRank) {
		//	System.out.println(c);
		//}

		if (change > allowedChange * numberOfCentroids) {
			return true;
		} else {
			return false;
		}
	}

	/**
	 * Calculates the kNN distance for the {@linkCentroid} by the given index. k
	 * does not indicates how many neighbouring {@linkCentroid}s have to be
	 * taken into account, but how big the accumulated weight of the
	 * neighbouring {@link Centroid}s has to be including the own weight.
	 * 
	 * @param centroids
	 *            The list of the {@link Centroid}s.
	 * @param index
	 *            The index of the {@link Centroid} for which the kNN distance
	 *            should be calculated.
	 * @param k
	 *            The accumulated weight of the neighbours have to have.
	 * @return The kNN distance, where k is the accumulated weight that has to
	 *         be reached.
	 */
	private double calculateKNNDistance(ArrayList<Centroid> centroids, int index, double k) {
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
				distSet.add(new DistanceObject(c.euclideanDistance(centroids.get(i).getVector()),
						centroids.get(i).getWeight()));
			}
		}

		double accumulatedWeight = centroids.get(index).getWeight();
		Iterator<DistanceObject> it = distSet.iterator();
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
