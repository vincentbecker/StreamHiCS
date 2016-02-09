package subspacebuilder;

import java.util.Collection;
import java.util.Set;
import org.jgrapht.UndirectedGraph;
import org.jgrapht.alg.BronKerboschCliqueFinder;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleGraph;

import fullsystem.Contrast;
import streamdatastructures.CorrelationSummary;
import subspace.Subspace;
import subspace.SubspaceSet;

public class CliqueBuilder extends SubspaceBuilder {

	private BronKerboschCliqueFinder<Integer, DefaultEdge> cliqueFinder;

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
	public CliqueBuilder(int numberOfDimensions, double threshold, Contrast contrastEvaluator,
			CorrelationSummary correlationSummary) {
		this.numberOfDimensions = numberOfDimensions;
		this.threshold = threshold;
		this.contrastEvaluator = contrastEvaluator;
		this.correlationSummary = correlationSummary;
	}

	@Override
	public SubspaceSet buildCorrelatedSubspaces() {
		double contrast = 0;
		// Create all 2-dimensional candidates
		// stopwatch.start("2D-contrast");

		UndirectedGraph<Integer, DefaultEdge> graph = new SimpleGraph<Integer, DefaultEdge>(DefaultEdge.class);
		Integer[] vertices = new Integer[numberOfDimensions];
		for (int i = 0; i < numberOfDimensions; i++) {
			Integer integer = new Integer(i);
			vertices[i] = integer;
			graph.addVertex(integer);
		}
		if (correlationSummary != null) {
			double[][] coefficientMatrix = correlationSummary.getCorrelationMatrix();
			for (int i = 0; i < numberOfDimensions - 1; i++) {
				for (int j = i + 1; j < numberOfDimensions; j++) {
					if (coefficientMatrix[i][j] >= threshold) {
						Subspace s = new Subspace();
						s.addDimension(i);
						s.addDimension(j);
						contrast = contrastEvaluator.evaluateSubspaceContrast(s);
						if (contrast >= threshold) {
							graph.addEdge(vertices[i], vertices[j]);
						}
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
					if (contrast >= threshold) {
						graph.addEdge(vertices[i], vertices[j]);
					}
				}
			}
		}

		// Find the maximal cliques
		this.cliqueFinder = new BronKerboschCliqueFinder<Integer, DefaultEdge>(graph);
		Collection<Set<Integer>> maximalCliques = cliqueFinder.getAllMaximalCliques();
		SubspaceSet correlatedSubspaces = new SubspaceSet();
		for (Set<Integer> clique : maximalCliques) {
			if(clique.size() > 1){
				Subspace s = new Subspace();
				for (Integer dimension : clique) {
					s.addDimension(dimension);
				}
				correlatedSubspaces.addSubspace(s);
			}
		}
		return correlatedSubspaces;
	}

}
