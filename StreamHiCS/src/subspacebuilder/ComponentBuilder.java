package subspacebuilder;

import java.util.ArrayDeque;

import fullsystem.Contrast;
import streamdatastructures.CorrelationSummary;
import subspace.Subspace;
import subspace.SubspaceSet;

/**
 * This class represents a {@link SubspaceBuilder}. The dimensions are seen as
 * vertices in a graph and if a two-dimensional subspace's contrast exceeds the
 * threshold an edge is created for the corresponding vertices. Afterwards all
 * connected sets of vertices (dimensions) are searched in the graph. These are
 * returned as correlated subspaces.
 * 
 * @author Vincent
 *
 */
public class ComponentBuilder extends SubspaceBuilder {

	/**
	 * The number of dimensions of the full space.
	 */
	private int numberOfDimensions;

	/**
	 * The minimum contrast value a {@link Subspace} must have to be a candidate
	 * for the correlated subspaces. Note that, even if a subspace's contrast
	 * exceeds the threshold it might not be chosen due to the cutoff. The
	 * threshold must be positive.
	 */
	private double threshold;

	//private int cutoff;

	/**
	 * The {@link Contrast} evaluator.
	 */
	private Contrast contrastEvaluator;

	/**
	 * The {@link CorrelationSummary} to calculate the Pearsons's correlation
	 * coefficient for pairs of dimensions.
	 */
	private CorrelationSummary correlationSummary;
	// private Stopwatch stopwatch;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param numberOfDimensions
	 *            The number of dimensions of the full space
	 * @param threshold
	 *            The threshold
	 * @param contrastEvaluator
	 *            The {@link Contrast} instance
	 * @param correlationSummary
	 *            The {@link CorrelationSummary}
	 */
	public ComponentBuilder(int numberOfDimensions, double threshold, Contrast contrastEvaluator,
			CorrelationSummary correlationSummary) {
		this.numberOfDimensions = numberOfDimensions;
		this.threshold = threshold;
		//this.cutoff = cutoff;
		this.contrastEvaluator = contrastEvaluator;
		this.correlationSummary = correlationSummary;
	}

	@Override
	public SubspaceSet buildCorrelatedSubspaces() {
		double contrast = 0;
		// Create all 2-dimensional candidates
		// stopwatch.start("2D-contrast");

		//SubspaceSet set = new SubspaceSet();
		double[][] adjacencyMatrix = new double[numberOfDimensions][numberOfDimensions];
		if (correlationSummary != null) {
			double[][] coefficientMatrix = correlationSummary.getCorrelationMatrix();
			for (int i = 0; i < numberOfDimensions - 1; i++) {
				for (int j = i + 1; j < numberOfDimensions; j++) {
					if (coefficientMatrix[i][j] >= threshold) {
						Subspace s = new Subspace();
						s.addDimension(i);
						s.addDimension(j);
						contrast = contrastEvaluator.evaluateSubspaceContrast(s);
						// if (contrast >= threshold) {
						//set.addSubspace(s);
						adjacencyMatrix[i][j] = contrast;
						adjacencyMatrix[j][i] = contrast;
						// }
					}
				}
			}
		} else {
			for (int i = 0; i < numberOfDimensions - 1; i++) {
				for (int j = i + 1; j < numberOfDimensions; j++) {
					Subspace s = new Subspace();
					s.addDimension(i);
					s.addDimension(j);
					// Only use subspaces for the further process which are
					// correlated
					contrast = contrastEvaluator.evaluateSubspaceContrast(s);
					s.setContrast(contrast);
					// if (contrast >= threshold) {
					adjacencyMatrix[i][j] = contrast;
					adjacencyMatrix[j][i] = contrast;
					//set.addSubspace(s);
					// }
				}
			}
		}
		/*
		set.selectTopK(cutoff);
		for (Subspace s : set.getSubspaces()) {
			int dim1 = s.getDimension(0);
			int dim2 = s.getDimension(1);
			adjacencyMatrix[dim1][dim2] = 1;
		}
	*/
		// Finding the connected components using depth-first search
		SubspaceSet correlatedSubspaces = new SubspaceSet();
		boolean[] visited = new boolean[numberOfDimensions];
		ArrayDeque<Integer> queue = new ArrayDeque<Integer>();
		int nextStart = 0;
		int dim;
		while (nextStart >= 0) {
			visited[nextStart] = true;
			Subspace s = new Subspace();
			s.addDimension(nextStart);
			queue.add(nextStart);
			while (!queue.isEmpty()) {
				dim = queue.remove();
				s.addDimension(dim);
				visited[dim] = true;
				for (int j = 0; j < numberOfDimensions; j++) {
					if (dim != j && adjacencyMatrix[dim][j] >= threshold && !visited[j]) {
						queue.add(j);
					}
				}
			}
			if (s.size() > 1) {
				contrast = contrastEvaluator.evaluateSubspaceContrast(s);
				s.setContrast(contrast);
				correlatedSubspaces.addSubspace(s);
			}
			/*
			 * if(s.size() == 2 && s.getContrast() < threshold){
			 * System.out.println("Hello"); }
			 */
			nextStart = nextVisit(visited);
		}

		return correlatedSubspaces;
	}

	private int nextVisit(boolean[] visited) {
		for (int i = 0; i < visited.length; i++) {
			if (!visited[i]) {
				return i;
			}
		}
		return -1;
	}
}
