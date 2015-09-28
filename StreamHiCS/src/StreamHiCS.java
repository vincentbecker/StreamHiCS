import java.util.ArrayList;
import java.util.Random;

import StatisticalTests.StatisticalTest;
import StatisticalTests.WelchTtest;
import StreamDataStructures.DataStreamContainer;
import StreamDataStructures.SlidingWindow;
import StreamDataStructures.Subspace;
import weka.core.Instance;

public class StreamHiCS {

	/**
	 * The number of {@link Instance} that are observed before the
	 * {@link Subspace} contrasts are checked again.
	 */
	private int updateInterval;
	/**
	 * To count the number of {@link Instance}s observed since the last
	 * {@link Subspace} evaluation.
	 */
	private int currentCount = 0;
	private ArrayList<Subspace> correlatedSubspaces;
	/**
	 * Data structure holding the {@link Instance}s.
	 */
	private DataStreamContainer dataStreamContainer;
	/**
	 * Number of Monte Carlo iterations in the contrast evaluation.
	 */
	private int m;
	/**
	 * The number of dimensions of the full space.
	 */
	private int numberOfDimensions;
	/**
	 * The relative size of the conditional (sliced) sample in relation to the
	 * whole data set.
	 */
	private double alpha;
	/**
	 * The contrast threshold. Every subspace of which the contrast exceeds the
	 * threshold is considered as correlated.
	 */
	private double threshold;
	/**
	 * The {@link StatisticalTest} used to calculate the deviation of the
	 * marginal sample and the conditional (sliced) sample.
	 */
	private StatisticalTest statisticalTest;
	/**
	 * A generator for random numbers.
	 */
	private Random generator;

	/**
	 * Creates a {@link StreamHiCS} object with the specified update interval.
	 * 
	 * @param updateInterval
	 *            The number how many {@link Instance}s are observed between
	 *            evaluations of the correlated {@link Subspace}s.
	 */
	public StreamHiCS(int numberOfDimensions, int updateInterval, double alpha,
			double threshold) {
		this.updateInterval = updateInterval;
		correlatedSubspaces = new ArrayList<Subspace>();
		dataStreamContainer = new SlidingWindow(numberOfDimensions, 100);
		this.numberOfDimensions = numberOfDimensions;
		this.alpha = alpha;
		this.threshold = threshold;
		// Try out other tests
		statisticalTest = new WelchTtest();
		generator = new Random();
	}

	/**
	 * Add a new {@link Instance}. If the number of observed {@link Instance}s
	 * since the last evaluation of the correlated {@link Subspace}s exceeds the
	 * update interval, the a new evaluation is carried out.
	 * 
	 * @param instance
	 *            The {@link Instance} to be added.
	 */
	public void add(Instance instance) {
		dataStreamContainer.add(instance);
		if (currentCount >= updateInterval) {
			evaluateCorrelatedSubspaces();
			currentCount = 0;
		}
	}

	/**
	 * Returns the number of {@link Instance}s currently contained in this
	 * object.
	 * 
	 * @return The number of {@link Instance}s currently contained in this
	 *         object.
	 */
	public int getNumberOfInstances() {
		return dataStreamContainer.getNumberOfInstances();
	}

	/**
	 * Carries out an evaluation of the stored correlated {@link Subspace}s and
	 * searches for new ones.
	 */
	private void evaluateCorrelatedSubspaces() {
		double contrast = 0;
		if (correlatedSubspaces.isEmpty()) {
			// Find new correlated subspaces
			buildCorrelatedSubspaces();
		}
		for (Subspace subspace : correlatedSubspaces) {
			contrast = evaluateSubspaceContrast(subspace);
			System.out.println(contrast);

			// If contrast has decreased in some subspaces we build new
			// subspaces.

		}
	}

	private void buildCorrelatedSubspaces() {
		ArrayList<Subspace> c_K = new ArrayList<Subspace>();
		// Create all 2-dimensional candidates
		for (int i = 0; i < numberOfDimensions; i++) {
			for (int j = 0; j < numberOfDimensions && i < j; j++) {
				Subspace s = new Subspace();
				s.addDimension(i);
				s.addDimension(j);
				c_K.add(s);
			}
		}

		// Carry out apriori algorithm
		apriori(c_K);
	}

	private void apriori(ArrayList<Subspace> c_K) {
		if (!c_K.isEmpty()) {
			correlatedSubspaces = c_K;
			ArrayList<Subspace> c_Kplus1 = new ArrayList<Subspace>();
			for (int i = 0; i < c_K.size(); i++) {
				for (int j = 0; j < c_K.size() && i < j; j++) {
					// Creating new candidates
					Subspace kPlus1Candidate = merge(c_K.get(i), c_K.get(j));

					if (kPlus1Candidate != null) {
						// Pruning
						if (checkCandidates(c_K, kPlus1Candidate)) {
							c_Kplus1.add(kPlus1Candidate);
						}
						c_Kplus1.add(kPlus1Candidate);
					}
				}
			}
			// Recurse
			apriori(c_Kplus1);
		}
	}

	private Subspace merge(Subspace s1, Subspace s2) {
		int k = s1.getSize();
		for (int i = 0; i < k - 1; i++) {
			if (s1.getDimension(i) != s2.getDimension(i)) {
				return null;
			}
		}
		s1.addDimension(s2.getDimension(k));
		return s1;

	}

	private boolean checkCandidates(ArrayList<Subspace> c_K, Subspace s) {
		// Does the candidate contain a subset, which is not correlated?
		Subspace sKminus1 = s.copy();
		for (int i = 0; i < s.getSize(); i++) {
			sKminus1.discardDimension(i);
			if (!c_K.contains(sKminus1)) {
				return false;
			}
			sKminus1.addDimension(i);
		}

		// Is the contrast higher or equal to the threshold?
		if (evaluateSubspaceContrast(s) < threshold) {
			return false;
		}
		return true;
	}

	/**
	 * Calculate the contrast for a given {@link Subspace}. See the HiCS paper
	 * for a description of the algorithm.
	 * 
	 * @param subspace
	 *            The {@link Subspace} for which the contrast should be
	 *            calculated.
	 * @return The contrast of the given {@link Subspace}.
	 */
	private double evaluateSubspaceContrast(Subspace subspace) {
		// Variable for collecting the intermediate results of the iterations
		double sum = 0;

		// Do Monte Carlo iterations
		for (int i = 0; i < m - 1; i++) {

			// Get random dimension
			int rnd = generator.nextInt(subspace.getSize());
			int dim = subspace.getDimension(rnd);

			// Calculate the number of instances selected per dimension
			double selectionAlpha = Math.pow(alpha, 1.0 / subspace.getSize());
			// Get the projected data
			double[] dimProjectedData = dataStreamContainer
					.getProjectedData(dim);
			// Get the randomly sliced data
			double[] slicedData = dataStreamContainer.getSlicedData(subspace,
					dim, selectionAlpha);
			// Calculate the deviation and add it to the overall sum
			sum += statisticalTest.calculateDeviation(dimProjectedData,
					slicedData);
		}

		// Return the mean of the intermediate results
		return sum / m;
	}
}
