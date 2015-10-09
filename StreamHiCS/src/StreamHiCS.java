import java.util.ArrayList;

import org.apache.commons.math3.util.MathArrays;
import statisticalTests.KolmogorovSmirnov;
import statisticalTests.StatisticalTest;
import streamDataStructures.DataStreamContainer;
import streamDataStructures.SlidingWindow;
import streamDataStructures.Subspace;
import streamDataStructures.SubspaceSet;
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
	/**
	 * The set of the currently correlated {@link Subspace}s.
	 */
	private SubspaceSet correlatedSubspaces;
	/**
	 * Data structure holding the {@link Instance}s.
	 */
	private DataStreamContainer dataStreamContainer;
	/**
	 * Number of Monte Carlo iterations in the contrast evaluation. m must be
	 * positive.
	 */
	private int m;
	/**
	 * The number of dimensions of the full space.
	 */
	private int numberOfDimensions;
	/**
	 * The relative size of the conditional (sliced) sample in relation to the
	 * whole data set. The number must be positive.
	 */
	private double alpha;
	/**
	 * Determines how much the contrast values of {@link Subspace]s are allowed
	 * to deviate from the last evaluation. If the deviation exceeds epsilon the
	 * correlated {Subspace}s are newly built. Epsilon must be positive.
	 */
	private double epsilon;
	/**
	 * The minimum contrast value a {@link Subspace} must have to be a candidate
	 * for the correlated subspaces. Note that, even if a subspace's contrast
	 * exceeds the threshold it might not be chosen due to the cutoff.
	 */
	private double threshold;
	/**
	 * The number of subspace candidates should be kept after each apriori step.
	 * The cutoff value must be positive. The threshold must be positive.
	 */
	private int cutoff;
	/**
	 * The {@link StatisticalTest} used to calculate the deviation of the
	 * marginal sample and the conditional (sliced) sample.
	 */
	private StatisticalTest statisticalTest;

	/**
	 * Creates a {@link StreamHiCS} object with the specified update interval.
	 * 
	 * @param numberOfDimensions
	 *            The number of dimensions of the full space.
	 * @param updateInterval
	 *            The number how many {@link Instance}s are observed between
	 *            evaluations of the correlated {@link Subspace}s.
	 * @param m
	 *            The number of Monte Carlo iterations for the estimation of the
	 *            conditional density.
	 * @param alpha
	 *            The fraction of data that should be selected in the estimation
	 *            of the conditional density.
	 * @param epsilon
	 *            The deviation that is allowed for contrast values between two
	 *            evaluation without starting a full new build of the correlated
	 *            subspaces.
	 * @param threshold
	 *            The threshold for the contrast. {@link Subspace}s with a
	 *            contrast above or equal to the threshold may be considered as
	 *            correlated.
	 */
	public StreamHiCS(int numberOfDimensions, int updateInterval, int m, double alpha, double epsilon, double threshold,
			int cutoff) {
		correlatedSubspaces = new SubspaceSet();
		// dataStreamContainer = new SelfOrganizingMap(numberOfDimensions, 100);
		dataStreamContainer = new SlidingWindow(numberOfDimensions, 10000);
		if (numberOfDimensions <= 0 || updateInterval <= 0 || m <= 0 || alpha <= 0 || epsilon <= 0 || cutoff <= 0) {
			throw new IllegalArgumentException("Non-positive input value.");
		}
		this.numberOfDimensions = numberOfDimensions;
		this.updateInterval = updateInterval;
		this.m = m;
		this.alpha = alpha;
		this.epsilon = epsilon;
		this.threshold = threshold;
		this.cutoff = cutoff;
		// Try out other tests
		statisticalTest = new KolmogorovSmirnov();
	}

	/**
	 * Returns the correlated {@link Subspace}s.
	 * 
	 * @return The {@link SubspaceSet} containing the currently correlated
	 *         {@link Subspace}s.
	 */
	public SubspaceSet getCurrentlyCorrelatedSubspaces() {
		return correlatedSubspaces;
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
		currentCount++;
		if (currentCount >= updateInterval) {
			evaluateCorrelatedSubspaces();
			currentCount = 0;
			System.out.println("Correlated: " + correlatedSubspaces.toString());
		}
	}

	/**
	 * Clears all stored {@link Instance}s.
	 */
	public void clear() {
		dataStreamContainer.clear();
		currentCount = 0;
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
		if (correlatedSubspaces.isEmpty()) {
			// Find new correlated subspaces
			buildCorrelatedSubspaces();
		} else {
			double contrast = 0;
			boolean update = false;
			// This variable is needed to build a workaround of problems with
			// the iterator when removing elements at the same time
			int l = correlatedSubspaces.size();
			for (int i = 1; i < l && !update; i++) {
				Subspace subspace = correlatedSubspaces.getSubspace(i);
				contrast = evaluateSubspaceContrast(subspace);
				// System.out.println(contrast);

				// If contrast has changed more than epsilon or has fallen below
				// the threshold we start a new
				// complete
				// evaluation.
				if (Math.abs(contrast - subspace.getContrast()) <= epsilon || contrast < threshold) {
					update = true;
					break;
					// l--;
				}
			}
			// If a subspace has changed we should update the correlated
			// subspaces.
			if (update) {
				correlatedSubspaces.clear();
				buildCorrelatedSubspaces();
			}
		}

		/*
		 * // Parallel version List<Double> res =
		 * correlatedSubspaces.parallelStream().map(s -> { return
		 * evaluateSubspaceContrast(s); }).collect(Collectors.toList());
		 * 
		 * for (Double d : res) { if (d < threshold) { update = true; } }
		 */
	}

	/**
	 * Builds a new set of correlated subspaces. The old ones are kept if they
	 * still are correlated.
	 */
	private void buildCorrelatedSubspaces() {
		SubspaceSet c_K = new SubspaceSet();
		double contrast = 0;
		// Create all 2-dimensional candidates
		for (int i = 0; i < numberOfDimensions - 1; i++) {
			for (int j = i + 1; j < numberOfDimensions; j++) {
				Subspace s = new Subspace();
				s.addDimension(i);
				s.addDimension(j);
				// Only use subspace for the further process which are
				// correlated
				contrast = evaluateSubspaceContrast(s);
				s.setContrast(contrast);
				if (contrast >= threshold) {
					c_K.addSubspace(s);
				}
			}
		}

		// Select cutoff subspaces
		c_K.selectTopK(cutoff);

		// Add the left over 2D subspaces to the correlated subspaces
		correlatedSubspaces.addSubspaces(c_K);

		// Carry out apriori algorithm
		apriori(c_K);
		//aprioriParallel(c_K);

		// Carry out pruning as the last step. All those subspaces which are
		// subspace to another subspace with higher contrast are discarded.
		prune();
	}

	/**
	 * Carries out the apriori-algorithm recursively.
	 * 
	 * @param c_K
	 *            The current candidate set for correlated subspaces in the
	 *            recursion.
	 */
	private void apriori(SubspaceSet c_K) {
		SubspaceSet c_Kplus1 = new SubspaceSet();
		c_K.sort();
		// The subspaces in the set are sorted. To speed up the process we can
		// stop looking
		// for a merge as soon as we have found the first "no merge" case
		boolean continueMerging = true;
		double contrast = 0;
		for (int i = 0; i < c_K.size() - 1; i++) {
			continueMerging = true;
			for (int j = i + 1; j < c_K.size() && continueMerging; j++) {
				// Creating new candidates
				Subspace kPlus1Candidate = Subspace.merge(c_K.getSubspace(i), c_K.getSubspace(j));

				if (kPlus1Candidate != null) {
					// Calculate the contrast of the subspace
					contrast = evaluateSubspaceContrast(kPlus1Candidate);
					kPlus1Candidate.setContrast(contrast);
					if (contrast >= threshold) {
						c_Kplus1.addSubspace(kPlus1Candidate);
					}
				} else {
					continueMerging = false;
				}
			}
		}
		if (!c_Kplus1.isEmpty()) {
			// Select the subspaces with highest contrast
			c_Kplus1.selectTopK(cutoff);
			correlatedSubspaces.addSubspaces(c_Kplus1);
			// Recurse
			apriori(c_Kplus1);
		}

		// TODO: Parallel excecution? At least checking procedure -> forEach()
	}

	/**
	 * Parallel version.
	 * 
	 * @param c_K
	 */
	private void aprioriParallel(SubspaceSet c_K) {
		SubspaceSet c_Kplus1 = new SubspaceSet();
		SubspaceSet temp = new SubspaceSet();
		c_K.sort();
		// The subspaces in the set are sorted. To speed up the process we can
		// stop looking
		// for a merge as soon as we have found the first "no merge" case
		boolean continueMerging = true;
		for (int i = 0; i < c_K.size() - 1; i++) {
			continueMerging = true;
			for (int j = i + 1; j < c_K.size() && continueMerging; j++) {
				// Creating new candidates
				Subspace kPlus1Candidate = Subspace.merge(c_K.getSubspace(i), c_K.getSubspace(j));
				if (kPlus1Candidate != null) {
					temp.addSubspace(kPlus1Candidate);
				} else {
					continueMerging = false;
				}
			}
		}

		// Parallel execution of the contrast calculation
		temp.getSubspaces().parallelStream().forEach(candidate -> {
			candidate.setContrast(evaluateSubspaceContrast(candidate));

		});

		// Add the subspaces greater or equal to the threshold to c_Kplus1
		for (Subspace candidate : temp.getSubspaces()) {
			if (candidate.getContrast() >= threshold) {
				c_Kplus1.addSubspace(candidate);
			}
		}

		if (!c_Kplus1.isEmpty()) {
			// Select the subspaces with highest contrast
			c_Kplus1.selectTopK(cutoff);
			correlatedSubspaces.addSubspaces(c_Kplus1);
			// Recurse
			aprioriParallel(c_Kplus1);
		}
	}

	/**
	 * If a {@link Subspace} is a subspace of another subspace with a higher
	 * contrast value, then it is discarded.
	 */
	private void prune() {
		ArrayList<Integer> discard = new ArrayList<Integer>();
		int l = correlatedSubspaces.size();
		Subspace si;
		Subspace sj;
		boolean discarded;
		for (int i = 0; i < l; i++) {
			discarded = false;
			si = correlatedSubspaces.getSubspace(i);
			for (int j = 0; j < l && !discarded; j++) {
				if (i != j) {
					sj = correlatedSubspaces.getSubspace(j);
					// If the correlated subspace contains a superset that has
					// at least (nearly) the same contrast we discard the
					// current subspace
					if (si.isSubspaceOf(sj) && si.getContrast() <= (sj.getContrast() + 0.01)) {
						discard.add(i);
						discarded = true;
					}
				}
			}
		}
		SubspaceSet prunedSet = new SubspaceSet();
		for (int i = 0; i < l; i++) {
			if (!discard.contains(i)) {
				prunedSet.addSubspace(correlatedSubspaces.getSubspace(i));
			}
		}
		correlatedSubspaces = prunedSet;
	}

	/**
	 * Checks if a candidate for a correlated subspace is a correlated subspace.
	 * 
	 * @param c_K
	 *            The candidate set.
	 * @param s
	 *            the {@link Subspace}.
	 * @return True is the subspace is correlated, false otherwise.
	 */
	private boolean checkCandidates(SubspaceSet c_K, Subspace s) {
		/*
		 * Formally, we are not allowed to apply apriori monocity principles.
		 * 
		 * 
		 * // If a candidate is a subset of a subspace which is correlated //
		 * (contrast above or equal to threshold), then the candidate is //
		 * correlated, too.
		 * 
		 * 
		 * // Does the candidate contain a subset, which is not correlated?
		 * Subspace sKminus1 = s.copy(); for (int i = 0; i < s.getSize(); i++) {
		 * sKminus1.discardDimension(i); if (!c_K.contains(sKminus1)) { return
		 * false; } sKminus1.addDimension(i); }
		 */

		// Is the contrast higher or equal to the threshold?
		// if (evaluateSubspaceContrast(s) < threshold) {
		// return false;
		// }
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
	public double evaluateSubspaceContrast(Subspace subspace) {
		// TODO: Parallel execution? But in higher level

		// Variable for collecting the intermediate results of the iterations
		double sum = 0;
		// A deviation could be NaN, so we wont count that calculation
		int numberOfCorrectTests = 0;
		int[] shuffledDimensions;
		double selectionAlpha = Math.pow(alpha, 1.0 / (subspace.size() - 1));
		double deviation;
		// Do Monte Carlo iterations
		for (int i = 0; i < m; i++) {
			shuffledDimensions = subspace.getDimensions();
			// Shuffle dimensions
			MathArrays.shuffle(shuffledDimensions);
			// Calculate the number of instances selected per dimension
			
			// Get the projected data
			double[] dimProjectedData = dataStreamContainer
					.getProjectedData(shuffledDimensions[shuffledDimensions.length - 1]);
			// Get the randomly sliced data
			// double[] slicedData = dataStreamContainer.getSlicedData(subspace,
			// dim, selectionAlpha);
			double[] slicedData = dataStreamContainer.getSlicedData(shuffledDimensions, selectionAlpha);
			// Calculate the deviation and add it to the overall sum
			deviation = statisticalTest.calculateDeviation(dimProjectedData, slicedData);
			if (!Double.isNaN(deviation)) {
				sum += deviation;
				numberOfCorrectTests++;
			}
		}

		// Return the mean of the intermediate results. If all results were NaN,
		// 0 is returned.
		double mean = 0;
		if (numberOfCorrectTests > 0) {
			mean = sum / numberOfCorrectTests;
		}
		return mean;
	}

	/**
	 * Returns a string representation of this object.
	 * 
	 * @return A string representation of this object.
	 */
	public String toString() {
		return correlatedSubspaces.toString();
	}
}