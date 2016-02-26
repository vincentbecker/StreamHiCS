package streamhics_hicstest;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.LinkedList;
import java.util.List;

import org.junit.BeforeClass;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import clustree.ClusTree;
import environment.CovarianceMatrixGenerator;
import environment.Evaluator;
import environment.Parameters.StreamSummarisation;
import environment.Parameters.SubspaceBuildup;
import environment.Stopwatch;
import fullsystem.Callback;
import fullsystem.Contrast;
import fullsystem.StreamHiCS;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.CorrelationSummary;
import streamdatastructures.SummarisationAdapter;
import streams.GaussianStream;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.HierarchicalBuilderCutoff;
import subspacebuilder.SubspaceBuilder;
import subspacebuilder.ComponentBuilder;
import weka.core.Instance;

public class HighDimRuntime {

	private GaussianStream stream;
	private StreamHiCS streamHiCS;
	private final int numInstances = 10000;
	private final int m = 50;
	private double alpha = 0.05;
	private double epsilon;
	private double threshold;
	private int cutoff;
	private double pruningDifference;
	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			// System.out.println("StreamHiCS: onAlarm()");
		}
	};
	private static Stopwatch stopwatch;
	private static final int numberTestRuns = 2;
	private List<String> results;

	@BeforeClass
	public static void setUpBeforeClass() {
		stopwatch = new Stopwatch();
	}

	@Test
	public void test1() {
		int horizon = 2000;
		double aprioriThreshold = 0.3;
		double hierarchicalThreshold = 0.45;
		double connectedComponentsThreshold = 0.5;
		boolean useCorrSummary = false;
		// SubspaceBuildup buildup = SubspaceBuildup.APRIORI;

		results = new LinkedList<String>();
		for (int b = 0; b < 2; b++) {
			if (b == 0) {
				useCorrSummary = false;
			} else {
				useCorrSummary = true;
			}
			
			results.add("Use correlation summary: " + useCorrSummary);

			for (SubspaceBuildup buildup : SubspaceBuildup.values()) {
				if (buildup == SubspaceBuildup.APRIORI || buildup == SubspaceBuildup.HIERARCHICAL
						|| buildup == SubspaceBuildup.CONNECTED_COMPONENTS) {
					if(useCorrSummary == true && buildup == SubspaceBuildup.HIERARCHICAL){				
					String summarisationDescription = null;
					String builderDescription = null;
					boolean addDescription = false;
					pruningDifference = 0.2;
					for (int d = 5; d <= 100; d += 5) {
						// double radius = 0.4 * Math.sqrt(d) + 0.1;
						double radius = 1;
						double learningRate = 1;
						// SummarisationAdapter adapter = new MCAdapter(horizon,
						// radius, learningRate, "radius");

						ClusTree clusTree = new ClusTree();
						clusTree.horizonOption.setValue(horizon);
						clusTree.prepareForUse();
						SummarisationAdapter adapter = new MicroclusteringAdapter(clusTree);

						if (summarisationDescription == null) {
							addDescription = true;
						}
						// summarisationDescription = "Radius centroids,
						// horizon: " + horizon + ", radius: " + radius
						// + ", learning rate: " + learningRate;
						summarisationDescription = "ClusTree depth-first, horizon: " + horizon;
						if (addDescription) {
							results.add(summarisationDescription);
						}
						Contrast contrastEvaluator = new Contrast(m, alpha, adapter);
						CorrelationSummary correlationSummary = null;
						if (useCorrSummary) {
							correlationSummary = new CorrelationSummary(d, horizon);
						}

						SubspaceBuilder subspaceBuilder = null;
						switch (buildup) {
						case APRIORI:
							threshold = aprioriThreshold;
							cutoff = 100;
							subspaceBuilder = new AprioriBuilder(d, threshold, cutoff, contrastEvaluator,
									correlationSummary);
							builderDescription = "Apriori, threshold: " + threshold + ", cutoff: " + cutoff
									+ ", correlationSummary: " + useCorrSummary;
							break;
						case HIERARCHICAL:
							threshold = hierarchicalThreshold;
							cutoff = 2;
							subspaceBuilder = new HierarchicalBuilderCutoff(d, threshold, cutoff, contrastEvaluator,
									correlationSummary, true);
							builderDescription = "Hierarchical, threshold: " + threshold + ", cutoff: " + cutoff
									+ ", correlationSummary: " + useCorrSummary;
							break;
						case CONNECTED_COMPONENTS:
							threshold = connectedComponentsThreshold;
							pruningDifference = -1;
							subspaceBuilder = new ComponentBuilder(d, threshold, contrastEvaluator, correlationSummary);
							builderDescription = "Connected components, threshold: " + threshold + ", correlationSummary: "
									+ useCorrSummary;
							break;
						}
						if (addDescription) {
							results.add(builderDescription);
							addDescription = false;
						}

						ChangeChecker changeChecker = new TimeCountChecker(numInstances);
						streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator,
								subspaceBuilder, changeChecker, callback, correlationSummary, stopwatch);
						changeChecker.setCallback(streamHiCS);

						int blockSize = d / 2;
						int[] blockBeginnings = { 0, blockSize };
						int[] blockSizes = { blockSize, blockSize };
						SubspaceSet correctResult = new SubspaceSet();
						Subspace s1 = new Subspace();
						Subspace s2 = new Subspace();
						for (int i = 0; i < blockSize; i++) {
							s1.addDimension(i);
							s2.addDimension(blockSize + i);
						}
						correctResult.addSubspace(s1);
						correctResult.addSubspace(s2);

						double[][] covarianceMatrix = CovarianceMatrixGenerator.generateCovarianceMatrix(d,
								blockBeginnings, blockSizes, 0.9);
						stream = new GaussianStream(null, covarianceMatrix, 1);

						double sumTPvsFP = 0;
						double sumAMJS = 0;
						double sumAMSS = 0;
						int sumElements = 0;

						stopwatch.reset();

						for (int i = 0; i < numberTestRuns; i++) {
							double[] performanceMeasures = testRun(correctResult);
							sumTPvsFP += performanceMeasures[0];
							sumAMJS += performanceMeasures[1];
							sumAMSS += performanceMeasures[2];
							sumElements += performanceMeasures[3];
						}

						// Calculate results
						double sumEvaluationTime = stopwatch.getTime("Evaluation");
						double sumAddingTime = stopwatch.getTime("Adding");
						double sumTotalTime = stopwatch.getTime("Total");

						double avgTPvsFP = sumTPvsFP / numberTestRuns;
						double avgAMJS = sumAMJS / numberTestRuns;
						double avgAMSS = sumAMSS / numberTestRuns;
						double avgNumElements = sumElements / numberTestRuns;
						double avgEvalTime = sumEvaluationTime / numberTestRuns;
						double avgAddingTime = sumAddingTime / numberTestRuns;
						double avgTotalTime = sumTotalTime / numberTestRuns;

						String measures = d + "," + avgTPvsFP + ", " + avgAMJS + ", " + avgAMSS + ", " + avgNumElements
								+ ", " + avgEvalTime + ", " + avgAddingTime + ", " + avgTotalTime;
						System.out.println(measures);
						results.add(measures);
					}
				}}
			}
		}

		// Write the results
		String filePath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/StreamHiCS/GaussianStreams/HighDimensional/Runtime_results.txt";

		try {
			Files.write(Paths.get(filePath), results, StandardOpenOption.APPEND);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private double[] testRun(SubspaceSet correctResult) {
		streamHiCS.clear();
		stream.restart();
		int numberSamples = 0;

		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			stopwatch.start("Total");
			streamHiCS.add(inst);
			stopwatch.stop("Total");
			numberSamples++;
		}

		SubspaceSet result = streamHiCS.getCurrentlyCorrelatedSubspaces();
		for (Subspace s : correctResult.getSubspaces()) {
			s.sort();
		}
		correctResult.sort();
		double[] performanceMeasures = new double[4];
		// Evaluator.displayResult(result, correctResult);
		performanceMeasures[0] = Evaluator.evaluateTPvsFP(result, correctResult);
		performanceMeasures[1] = Evaluator.evaluateJaccardIndex(result, correctResult);
		performanceMeasures[2] = Evaluator.evaluateStructuralSimilarity(result, correctResult);
		performanceMeasures[3] = streamHiCS.getNumberOfElements();

		return performanceMeasures;
	}

}
