package subspacebuilder;

import fullsystem.Contrast;
import streamdatastructures.CorrelationSummary;
import subspace.Subspace;
import subspace.SubspaceSet;

/**
 * This class represents a {@link SubspaceBuilder} using a top-down procedure to
 * find the correlated {@link Subspace}s. It starts with the full space and
 * recursively splits the {@link Subspace}s until correlated {@link Subspace}s
 * are found.
 * 
 * @author Vincent
 *
 */
public class HierarchicalBuilderCutoff extends SubspaceBuilder {

	/**
	 * A {@link SubspaceSet} containing the candidates for correlated
	 * {@link Subspace}s.
	 */
	private SubspaceSet correlatedSubspaces;

	/**
	 * A {@link SubspaceSet} containing the {@link Subspace}s which were
	 * assessed as not correlated.
	 */
	private SubspaceSet notCorrelatedSubspaces;

	/**
	 * The number of dimensions of the full space.
	 */
	private int numberOfDimensions;

	/**
	 * The minimum contrast value a {@link Subspace} must have to be a candidate
	 * for the correlated subspaces. THe threshold must be positive.
	 */
	private double threshold;

	/**
	 * The number of subspace candidates should be kept on each level of the
	 * tree. The cutoff value must be positive.
	 */
	private int cutoff;

	/**
	 * The @link{Contrast} instance.
	 */
	private Contrast contrastEvaluator;

	/**
	 * The {@link CorrelationSummary} to calculate the Pearsons's correlation
	 * coefficient for pairs of dimensions.
	 */
	private CorrelationSummary correlationSummary;

	/**
	 * Holds the contrast of all two-dimensional {@link Subspace}s.
	 */
	private double[][] contrastMatrix;

	/**
	 * A flag determining if a two {@link Subspace}s can overlap in dimensions
	 * or not.
	 */
	private boolean partition;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param numberOfDimensions
	 *            The number of dimensions of the full space
	 * @param threshold
	 *            The threshold
	 * @param contrastEvaluator
	 *            The {@link Contrast} instance
	 * @param partition
	 *            The flag determining whether to partition a {@link Subspace}
	 *            in the splitting process
	 */
	public HierarchicalBuilderCutoff(int numberOfDimensions, double threshold, int cutoff, Contrast contrastEvaluator,
			CorrelationSummary correlationSummary, boolean partition) {
		this.correlatedSubspaces = new SubspaceSet();
		this.notCorrelatedSubspaces = new SubspaceSet();
		this.numberOfDimensions = numberOfDimensions;
		this.threshold = threshold;
		this.cutoff = cutoff;
		this.contrastEvaluator = contrastEvaluator;
		this.correlationSummary = correlationSummary;
		this.partition = partition;
	}

	@Override
	public SubspaceSet buildCorrelatedSubspaces() {
		correlatedSubspaces.clear();
		notCorrelatedSubspaces.clear();

		if (correlationSummary != null) {
			contrastMatrix = correlationSummary.getCorrelationMatrix();
		} else {
			contrastMatrix = new double[numberOfDimensions][numberOfDimensions];
			double contrast = 0;
			// Calculate the contrast for all two dimensional subspaces and
			// store them in a lookup matrix since they are needed for
			// splitting.
			Subspace s;
			for (int i = 0; i < numberOfDimensions - 1; i++) {
				for (int j = i + 1; j < numberOfDimensions; j++) {
					s = new Subspace();
					s.addDimension(i);
					s.addDimension(j);
					contrast = contrastEvaluator.evaluateSubspaceContrast(s);
					s.setContrast(contrast);
					contrastMatrix[i][j] = contrast;
					contrastMatrix[j][i] = contrast;
				}
			}
		}

		// Create the full space
		Subspace fullSpace = new Subspace();
		boolean add = false;
		// TODO
		for (int i = 0; i < numberOfDimensions; i++) {
			add = false;
			for (int j = 0; j < numberOfDimensions && !add; j++) {
				if (contrastMatrix[i][j] >= threshold) {
					add = true;
				}
			}
			if (add) {
				fullSpace.addDimension(i);
			}
		}
		// System.out.println(fullSpace.toString());

		fullSpace.setContrast(contrastEvaluator.evaluateSubspaceContrast(fullSpace));

		// Start the recursive procedure
		SubspaceSet c_K = new SubspaceSet();
		c_K.addSubspace(fullSpace);
		hierarchicalBuildup(c_K);

		return correlatedSubspaces;
	}

	private void hierarchicalBuildup(SubspaceSet c_K) {
		SubspaceSet c_Kplus1 = new SubspaceSet();
		int size = 0;
		for (Subspace s : c_K.getSubspaces()) {
			if (s.getContrast() >= threshold) {
				correlatedSubspaces.addSubspace(s);
			}
			size = s.size();
			if (size > 2) {
				if (partition || !correlatedSubspaces.contains(s) && !notCorrelatedSubspaces.contains(s)) {

					if (s.getContrast() >= threshold) {
						correlatedSubspaces.addSubspace(s);
					} else {
						SubspaceSet splittingResult = split(s);
						System.out.println(splittingResult.toString());
						for (Subspace sr : splittingResult.getSubspaces()) {
							sr.setContrast(contrastEvaluator.evaluateSubspaceContrast(sr));
							if (sr.size() > 1) {
								c_Kplus1.addSubspace(sr);
							}
						}
					}
				}
			}
		}

		if (!c_Kplus1.isEmpty()) {
			// Select the subspaces with highest contrast
			c_Kplus1.selectTopK(cutoff);
			// System.out.println(s.toString() + " : " + contrast);
			// Recurse
			hierarchicalBuildup(c_Kplus1);
		}
	}

	/**
	 * Split a {@link Subspace} by using the two dimensions with the lowest
	 * common contrast as seeds for the two new {@link Subspace}s.
	 * 
	 * @param s
	 *            The {@link Subspace} to be split
	 * @return A {@link SubspaceSet} containing the two new {@link Subspace}s.
	 */
	private SubspaceSet split(Subspace s) {
		int size = s.size();
		double minContrast = Double.MAX_VALUE;
		double contrast = 0;
		int minI = 0;
		int minJ = 0;
		for (int i = 0; i < size; i++) {
			for (int j = i + 1; j < size; j++) {
				contrast = contrastMatrix[s.getDimension(i)][s.getDimension(j)];
				if (contrast < minContrast) {
					minContrast = contrast;
					minI = i;
					minJ = j;
				}
			}
		}
		// minI and minJ are the splitting seeds
		SubspaceSet result = new SubspaceSet();
		Subspace sI = new Subspace();
		Subspace sJ = new Subspace();
		sI.addDimension(s.getDimension(minI));
		sJ.addDimension(s.getDimension(minJ));

		if (partition) {
			// We form new subspaces by adding dimensions to the seed which they
			// share a higher two dimensional contrast with. That means the
			// seeds attract the dimensions by contrast
			double iContrast = 0;
			double jContrast = 0;
			int iDim = s.getDimension(minI);
			int jDim = s.getDimension(minJ);
			int dim;
			for (int i = 0; i < s.size(); i++) {
				if (i != minI && i != minJ) {
					dim = s.getDimension(i);
					iContrast = contrastMatrix[iDim][dim];
					jContrast = contrastMatrix[jDim][dim];
					if (iContrast >= jContrast) {
						sI.addDimension(dim);
					} else {
						sJ.addDimension(dim);
					}
				}
			}
		} else {
			// We form two new subspaces by adding adding one seed to either of
			// them and add all the other dimensions
			int dim = 0;
			for (int i = 0; i < s.size(); i++) {
				if (i != minI && i != minJ) {
					dim = s.getDimension(i);
					sI.addDimension(dim);
					sJ.addDimension(dim);
				}
			}
		}
		result.addSubspace(sI);
		result.addSubspace(sJ);
		return result;
	}
}
