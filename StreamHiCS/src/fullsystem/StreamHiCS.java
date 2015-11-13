package fullsystem;

import java.util.ArrayList;

import changechecker.ChangeChecker;
import contrast.Contrast;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class StreamHiCS implements Callback {
	/**
	 * The set of the currently correlated {@link Subspace}s.
	 */
	private SubspaceSet correlatedSubspaces;
	/**
	 * Determines how much the contrast values of {@link Subspace]s are allowed
	 * to deviate from the last evaluation. If the deviation exceeds epsilon the
	 * correlated {Subspace}s are newly built. Epsilon must be positive.
	 */
	private double epsilon;
	/**
	 * The minimum contrast value a {@link Subspace} must have to be a candidate
	 * for the correlated subspaces.
	 */
	private double threshold;
	/**
	 * The difference in contrast allowed to prune a {@link Subspace}.
	 */
	private double pruningDifference;
	/**
	 * The @link{Contrast} evaluator.
	 */
	private Contrast contrastEvaluator;
	private SubspaceBuilder subspaceBuilder;
	private ChangeChecker changeChecker;
	/**
	 * The @link{Callback} to notify on changes.
	 */
	private Callback callback;

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
	public StreamHiCS(double epsilon, double threshold, double pruningDifference, Contrast contrastEvaluator, SubspaceBuilder subspaceBuilder, ChangeChecker changeChecker,
			Callback callback) {
		correlatedSubspaces = new SubspaceSet();
		if (epsilon < 0) {
			throw new IllegalArgumentException("Non-positive input value.");
		}
		this.epsilon = epsilon;
		this.threshold = threshold;
		this.pruningDifference = pruningDifference;
		this.contrastEvaluator = contrastEvaluator;
		this.subspaceBuilder = subspaceBuilder;
		this.changeChecker = changeChecker;
		this.callback = callback;
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
		changeChecker.poll();
	}

	/**
	 * Clears all stored information learnt from the stream.
	 */
	public void clear() {
		correlatedSubspaces.clear();
		contrastEvaluator.clear();
	}

	/**
	 * Carries out an evaluation of the stored correlated {@link Subspace}s and
	 * searches for new ones.
	 */
	private boolean evaluateCorrelatedSubspaces() {
		boolean update = false;
		if (correlatedSubspaces.isEmpty()) {
			// Find new correlated subspaces
			correlatedSubspaces = subspaceBuilder.buildCorrelatedSubspaces();
			prune();
			if (!correlatedSubspaces.isEmpty()) {
				update = true;
			}
		} else {
			double contrast = 0;
			update = false;
			// This variable is needed to build a workaround of problems with
			// the iterator when removing elements at the same time
			int l = correlatedSubspaces.size();
			SubspaceSet keep = new SubspaceSet();
			for (int i = 0; i < l && !update; i++) {
				Subspace subspace = correlatedSubspaces.getSubspace(i);
				contrast = contrastEvaluator.evaluateSubspaceContrast(subspace);
				// System.out.println(contrast);

				// If contrast has changed more than epsilon or has fallen below
				// the threshold we start a new complete evaluation.
				if (Math.abs(contrast - subspace.getContrast()) > epsilon || contrast < threshold) {
					update = true;
				}else{
					keep.addSubspace(subspace);
				}
				subspace.setContrast(contrast);
			}
			// If a subspace has changed we update the correlated
			// subspaces.
			if (update) {
				correlatedSubspaces.clear();
				correlatedSubspaces.addSubspaces(keep);
				correlatedSubspaces.addSubspaces(subspaceBuilder.buildCorrelatedSubspaces());
				// Carry out pruning as the last step. All those subspaces which are
				// subspace to another subspace with higher contrast are discarded.
				prune();
			}
		}

		return update;

		/*
		 * TODO: Remove // Parallel version List<Double> res =
		 * correlatedSubspaces.parallelStream().map(s -> { return
		 * evaluateSubspaceContrast(s); }).collect(Collectors.toList());
		 * 
		 * for (Double d : res) { if (d < threshold) { update = true; } }
		 */
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
	 * Returns a string representation of this object.
	 * 
	 * @return A string representation of this object.
	 */
	public String toString() {
		return correlatedSubspaces.toString();
	}

	@Override
	public void onAlarm() {
		if (evaluateCorrelatedSubspaces()) {
			// Notify the callback
			callback.onAlarm();
		}
	}
}