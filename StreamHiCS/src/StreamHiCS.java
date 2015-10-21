import java.util.ArrayList;

import contrast.Callback;
import contrast.Contrast;
import streamDataStructures.Subspace;
import streamDataStructures.SubspaceSet;
import weka.core.Instance;

public class StreamHiCS implements Callback {
	/**
	 * The set of the currently correlated {@link Subspace}s.
	 */
	private SubspaceSet correlatedSubspaces;
	/**
	 * The number of dimensions of the full space.
	 */
	private int numberOfDimensions;
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
	 * The difference in contrast allowed to prune a {@link Subspace}.
	 */
	private double pruningDifference;
	/**
	 * The @link{Contrast} evaluator.
	 */
	private Contrast contrastEvaluator;

	/**
	 * Creates a {@link StreamHiCS} object with the specified update interval.
	 * 
	 * @param numberOfDimensions
	 *            The number of dimensions of the full space.
	 * @param epsilon
	 *            The deviation that is allowed for contrast values between two
	 *            evaluation without starting a full new build of the correlated
	 *            subspaces.
	 * @param threshold
	 *            The threshold for the contrast. {@link Subspace}s with a
	 *            contrast above or equal to the threshold may be considered as
	 *            correlated.
	 */
	public StreamHiCS(int numberOfDimensions, double epsilon, double threshold, int cutoff, double pruningDifference,
			Contrast contrastEvaluator) {
		correlatedSubspaces = new SubspaceSet();
		if (numberOfDimensions <= 0 || epsilon < 0 || cutoff <= 0 || pruningDifference < 0) {
			throw new IllegalArgumentException("Non-positive input value.");
		}
		this.numberOfDimensions = numberOfDimensions;
		this.epsilon = epsilon;
		this.threshold = threshold;
		this.cutoff = cutoff;
		this.pruningDifference = pruningDifference;
		this.contrastEvaluator = contrastEvaluator;
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
	 * Add a new {@link Instance}.
	 * 
	 * @param instance
	 *            The {@link Instance} to be added.
	 */
	public void add(Instance instance) {
		contrastEvaluator.add(instance);
	}

	/**
	 * Clears all stored information learnt from the stream.
	 */
	public void clear() {
		contrastEvaluator.clear();
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
				contrast = contrastEvaluator.evaluateSubspaceContrast(subspace);
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
		
		System.out.println("Correlated: " + correlatedSubspaces.toString());
		if(!correlatedSubspaces.isEmpty()){
			for(Subspace s : correlatedSubspaces.getSubspaces()){
				System.out.print(s.getContrast() + ", ");
			}
			System.out.println();
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
				contrast = contrastEvaluator.evaluateSubspaceContrast(s);
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
		// aprioriParallel(c_K);

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
					contrast = contrastEvaluator.evaluateSubspaceContrast(kPlus1Candidate);
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
			candidate.setContrast(contrastEvaluator.evaluateSubspaceContrast(candidate));

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
					if (si.isSubspaceOf(sj) && si.getContrast() <= (sj.getContrast() + pruningDifference)) {
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
	 * Returns a string representation of this object.
	 * 
	 * @return A string representation of this object.
	 */
	public String toString() {
		return correlatedSubspaces.toString();
	}

	@Override
	public void onAlarm() {
		evaluateCorrelatedSubspaces();
	}
}