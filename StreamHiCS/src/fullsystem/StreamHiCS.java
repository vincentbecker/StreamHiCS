package fullsystem;

import contrast.Callback;
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
	 * The @link{Contrast} evaluator.
	 */
	private Contrast contrastEvaluator;
	private SubspaceBuilder subspaceBuilder;
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
	public StreamHiCS(double epsilon, double threshold, Contrast contrastEvaluator, SubspaceBuilder subspaceBuilder,
			Callback callback) {
		correlatedSubspaces = new SubspaceSet();
		if (epsilon < 0) {
			throw new IllegalArgumentException("Non-positive input value.");
		}
		this.epsilon = epsilon;
		this.threshold = threshold;
		this.contrastEvaluator = contrastEvaluator;
		this.subspaceBuilder = subspaceBuilder;
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
			update = true;
		} else {
			double contrast = 0;
			update = false;
			// This variable is needed to build a workaround of problems with
			// the iterator when removing elements at the same time
			int l = correlatedSubspaces.size();
			for (int i = 1; i < l && !update; i++) {
				Subspace subspace = correlatedSubspaces.getSubspace(i);
				contrast = contrastEvaluator.evaluateSubspaceContrast(subspace);
				// System.out.println(contrast);

				// If contrast has changed more than epsilon or has fallen below
				// the threshold we start a new complete evaluation.
				if (Math.abs(contrast - subspace.getContrast()) <= epsilon || contrast < threshold) {
					update = true;
					break;
				}
			}
			// If a subspace has changed we update the correlated
			// subspaces.
			if (update) {
				correlatedSubspaces.clear();
				correlatedSubspaces = subspaceBuilder.buildCorrelatedSubspaces();
			}
		}

		System.out.println("Correlated: " + correlatedSubspaces.toString());
		if (!correlatedSubspaces.isEmpty()) {
			for (Subspace s : correlatedSubspaces.getSubspaces()) {
				System.out.print(s.getContrast() + ", ");
			}
			System.out.println();
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