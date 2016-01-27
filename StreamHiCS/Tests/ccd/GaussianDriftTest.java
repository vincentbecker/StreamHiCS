package ccd;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.junit.BeforeClass;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import clustree.ClusTree;
import environment.CSVReader;
import environment.Evaluator;
import environment.Stopwatch;
import environment.Parameters.StreamSummarisation;
import environment.Parameters.SubspaceBuildup;
import fullsystem.Callback;
import fullsystem.Contrast;
import fullsystem.CorrelatedSubspacesChangeDetector;
import fullsystem.FullSpaceChangeDetector;
import fullsystem.StreamHiCS;
import moa.clusterers.clustream.Clustream;
import moa.streams.ConceptDriftStream;
import streamdatastructures.CentroidsAdapter;
import streamdatastructures.CorrelationSummary;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SlidingWindowAdapter;
import streamdatastructures.SummarisationAdapter;
import streamdatastructures.WithDBSCAN;
import streams.GaussianStream;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.HierarchicalBuilderCutoff;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class GaussianDriftTest {
	private ConceptDriftStream conceptDriftStream;
	private StreamHiCS streamHiCS;
	private final int numInstances = 10000;
	private final int horizon = 1000;
	private final int m = 50;
	private final double alpha = 0.05;
	private double epsilon = 0;
	private double aprioriThreshold;
	private double hierarchicalThreshold;
	private int cutoff;
	private double pruningDifference;
	private int numberOfDimensions;
	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			// System.out.println("StreamHiCS: onAlarm()");
		}
	};
	private static Stopwatch stopwatch;
	private Contrast contrastEvaluator;
	private static final int numberTestRuns = 10;
	private List<String> results;
	private String summarisationDescription = null;
	private String builderDescription = null;
	private static CSVReader csvReader;
	private static final String path = "Tests/CovarianceMatrices/";
	private double[][] resultSummary;
	private CorrelatedSubspacesChangeDetector cscd;
	private FullSpaceChangeDetector refDetector;

	@BeforeClass
	public static void setUpBeforeClass() {
		stopwatch = new Stopwatch();
		csvReader = new CSVReader();
	}

	@Test
	public void test() {
		// Output
		results = new LinkedList<String>();
		SummarisationAdapter adapter;
		SubspaceBuilder subspaceBuilder;
		for (StreamSummarisation summarisation : StreamSummarisation.values()) {
			summarisationDescription = null;
			for (SubspaceBuildup buildup : SubspaceBuildup.values()) {
				builderDescription = null;
				if (summarisation == StreamSummarisation.CLUSTREE_DEPTHFIRST && buildup == SubspaceBuildup.APRIORI) {
					resultSummary = new double[2][7];
					for (int test = 1; test <= 4; test++) {
						stopwatch.reset();
						double threshold = 1;
						switch (buildup) {
						case APRIORI:
							threshold = aprioriThreshold;
							break;
						case HIERARCHICAL:
							threshold = hierarchicalThreshold;
							break;
						}

						ArrayList<Double> trueChanges = new ArrayList<Double>();

						switch (test) {
						case 1:
							numberOfDimensions = 5;
							GaussianStream s1 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 1.0);

							GaussianStream s2 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 1.2);

							GaussianStream s3 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 1.4);

							GaussianStream s4 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 1.2);

							GaussianStream s5 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 1.0);

							ConceptDriftStream conceptDriftStream1 = new ConceptDriftStream();
							conceptDriftStream1.streamOption.setCurrentObject(s1);
							conceptDriftStream1.driftstreamOption.setCurrentObject(s2);
							conceptDriftStream1.positionOption.setValue(5000);
							conceptDriftStream1.widthOption.setValue(1000);
							conceptDriftStream1.prepareForUse();
							trueChanges.add(5000.0);

							ConceptDriftStream conceptDriftStream2 = new ConceptDriftStream();
							conceptDriftStream2.streamOption.setCurrentObject(conceptDriftStream1);
							conceptDriftStream2.driftstreamOption.setCurrentObject(s3);
							conceptDriftStream2.positionOption.setValue(10000);
							conceptDriftStream2.widthOption.setValue(1000);
							conceptDriftStream2.prepareForUse();
							trueChanges.add(10000.0);

							ConceptDriftStream conceptDriftStream3 = new ConceptDriftStream();
							conceptDriftStream2.streamOption.setCurrentObject(conceptDriftStream2);
							conceptDriftStream2.driftstreamOption.setCurrentObject(s4);
							conceptDriftStream2.positionOption.setValue(15000);
							conceptDriftStream2.widthOption.setValue(1000);
							conceptDriftStream2.prepareForUse();
							trueChanges.add(15000.0);

							conceptDriftStream = new ConceptDriftStream();
							conceptDriftStream.streamOption.setCurrentObject(conceptDriftStream3);
							conceptDriftStream.driftstreamOption.setCurrentObject(s5);
							conceptDriftStream.positionOption.setValue(20000);
							conceptDriftStream.widthOption.setValue(1000);
							conceptDriftStream.prepareForUse();
							trueChanges.add(20000.0);

							break;
						case 2:
							numberOfDimensions = 5;
							s1 = new GaussianStream(null, csvReader.read(path + "Test5.csv"), 1.0);

							s2 = new GaussianStream(null, csvReader.read(path + "Test5.csv"), 1.2);

							s3 = new GaussianStream(null, csvReader.read(path + "Test5.csv"), 1.4);

							s4 = new GaussianStream(null, csvReader.read(path + "Test5.csv"), 1.2);

							s5 = new GaussianStream(null, csvReader.read(path + "Test5.csv"), 1.0);

							conceptDriftStream1 = new ConceptDriftStream();
							conceptDriftStream1.streamOption.setCurrentObject(s1);
							conceptDriftStream1.driftstreamOption.setCurrentObject(s2);
							conceptDriftStream1.positionOption.setValue(5000);
							conceptDriftStream1.widthOption.setValue(1000);
							conceptDriftStream1.prepareForUse();
							trueChanges.add(5000.0);

							conceptDriftStream2 = new ConceptDriftStream();
							conceptDriftStream2.streamOption.setCurrentObject(conceptDriftStream1);
							conceptDriftStream2.driftstreamOption.setCurrentObject(s3);
							conceptDriftStream2.positionOption.setValue(10000);
							conceptDriftStream2.widthOption.setValue(1000);
							conceptDriftStream2.prepareForUse();
							trueChanges.add(10000.0);

							conceptDriftStream3 = new ConceptDriftStream();
							conceptDriftStream3.streamOption.setCurrentObject(conceptDriftStream2);
							conceptDriftStream3.driftstreamOption.setCurrentObject(s4);
							conceptDriftStream3.positionOption.setValue(15000);
							conceptDriftStream3.widthOption.setValue(1000);
							conceptDriftStream3.prepareForUse();
							trueChanges.add(15000.0);

							conceptDriftStream = new ConceptDriftStream();
							conceptDriftStream.streamOption.setCurrentObject(conceptDriftStream3);
							conceptDriftStream.driftstreamOption.setCurrentObject(s5);
							conceptDriftStream.positionOption.setValue(20000);
							conceptDriftStream.widthOption.setValue(1000);
							conceptDriftStream.prepareForUse();
							trueChanges.add(20000.0);
							break;
						case 3:
							numberOfDimensions = 5;
							s1 = new GaussianStream(null, csvReader.read(path + "Test1.csv"), 1.0);

							s2 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 1.0);

							s3 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 1.2);

							s4 = new GaussianStream(null, csvReader.read(path + "Test12.csv"), 1.2);

							s5 = new GaussianStream(null, csvReader.read(path + "Test12.csv"), 1.4);
							
							GaussianStream s6 = new GaussianStream(null, csvReader.read(path + "Test13.csv"), 1.6);
							
							GaussianStream s7 = new GaussianStream(null, csvReader.read(path + "Test13.csv"), 1.4);

							conceptDriftStream1 = new ConceptDriftStream();
							conceptDriftStream1.streamOption.setCurrentObject(s1);
							conceptDriftStream1.driftstreamOption.setCurrentObject(s2);
							conceptDriftStream1.positionOption.setValue(5000);
							conceptDriftStream1.widthOption.setValue(1000);
							conceptDriftStream1.prepareForUse();
							trueChanges.add(5000.0);

							conceptDriftStream2 = new ConceptDriftStream();
							conceptDriftStream2.streamOption.setCurrentObject(conceptDriftStream1);
							conceptDriftStream2.driftstreamOption.setCurrentObject(s3);
							conceptDriftStream2.positionOption.setValue(10000);
							conceptDriftStream2.widthOption.setValue(1000);
							conceptDriftStream2.prepareForUse();
							trueChanges.add(10000.0);

							conceptDriftStream3 = new ConceptDriftStream();
							conceptDriftStream3.streamOption.setCurrentObject(conceptDriftStream2);
							conceptDriftStream3.driftstreamOption.setCurrentObject(s4);
							conceptDriftStream3.positionOption.setValue(15000);
							conceptDriftStream3.widthOption.setValue(1000);
							conceptDriftStream3.prepareForUse();
							trueChanges.add(15000.0);
							
							ConceptDriftStream conceptDriftStream4 = new ConceptDriftStream();
							conceptDriftStream4.streamOption.setCurrentObject(conceptDriftStream3);
							conceptDriftStream4.driftstreamOption.setCurrentObject(s5);
							conceptDriftStream4.positionOption.setValue(20000);
							conceptDriftStream4.widthOption.setValue(1000);
							conceptDriftStream4.prepareForUse();
							trueChanges.add(20000.0);
							
							ConceptDriftStream conceptDriftStream5 = new ConceptDriftStream();
							conceptDriftStream5.streamOption.setCurrentObject(conceptDriftStream4);
							conceptDriftStream5.driftstreamOption.setCurrentObject(s6);
							conceptDriftStream5.positionOption.setValue(25000);
							conceptDriftStream5.widthOption.setValue(1000);
							conceptDriftStream5.prepareForUse();
							trueChanges.add(25000.0);

							conceptDriftStream = new ConceptDriftStream();
							conceptDriftStream.streamOption.setCurrentObject(conceptDriftStream5);
							conceptDriftStream.driftstreamOption.setCurrentObject(s7);
							conceptDriftStream.positionOption.setValue(30000);
							conceptDriftStream.widthOption.setValue(1000);
							conceptDriftStream.prepareForUse();
							trueChanges.add(30000.0);
							break;
						case 4:
							numberOfDimensions = 10;
							s1 = new GaussianStream(null, csvReader.read(path + "Test24.csv"), 1.0);

							s2 = new GaussianStream(null, csvReader.read(path + "Test24.csv"), 1.2);

							s3 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 1.2);

							s4 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 1.4);

							s5 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 1.2);
							
							s6 = new GaussianStream(null, csvReader.read(path + "Test26.csv"), 1.2);
							
							s7 = new GaussianStream(null, csvReader.read(path + "Test26.csv"), 1.4);
							
							GaussianStream s8 = new GaussianStream(null, csvReader.read(path + "Test28.csv"), 1.4);

							conceptDriftStream1 = new ConceptDriftStream();
							conceptDriftStream1.streamOption.setCurrentObject(s1);
							conceptDriftStream1.driftstreamOption.setCurrentObject(s2);
							conceptDriftStream1.positionOption.setValue(5000);
							conceptDriftStream1.widthOption.setValue(1000);
							conceptDriftStream1.prepareForUse();
							trueChanges.add(5000.0);

							conceptDriftStream2 = new ConceptDriftStream();
							conceptDriftStream2.streamOption.setCurrentObject(conceptDriftStream1);
							conceptDriftStream2.driftstreamOption.setCurrentObject(s3);
							conceptDriftStream2.positionOption.setValue(10000);
							conceptDriftStream2.widthOption.setValue(1000);
							conceptDriftStream2.prepareForUse();
							trueChanges.add(10000.0);

							conceptDriftStream3 = new ConceptDriftStream();
							conceptDriftStream3.streamOption.setCurrentObject(conceptDriftStream2);
							conceptDriftStream3.driftstreamOption.setCurrentObject(s4);
							conceptDriftStream3.positionOption.setValue(15000);
							conceptDriftStream3.widthOption.setValue(1000);
							conceptDriftStream3.prepareForUse();
							trueChanges.add(15000.0);
							
							conceptDriftStream4 = new ConceptDriftStream();
							conceptDriftStream4.streamOption.setCurrentObject(conceptDriftStream3);
							conceptDriftStream4.driftstreamOption.setCurrentObject(s5);
							conceptDriftStream4.positionOption.setValue(20000);
							conceptDriftStream4.widthOption.setValue(1000);
							conceptDriftStream4.prepareForUse();
							trueChanges.add(20000.0);
							
							conceptDriftStream5 = new ConceptDriftStream();
							conceptDriftStream5.streamOption.setCurrentObject(conceptDriftStream4);
							conceptDriftStream5.driftstreamOption.setCurrentObject(s6);
							conceptDriftStream5.positionOption.setValue(25000);
							conceptDriftStream5.widthOption.setValue(1000);
							conceptDriftStream5.prepareForUse();
							trueChanges.add(25000.0);
							
							ConceptDriftStream conceptDriftStream6 = new ConceptDriftStream();
							conceptDriftStream6.streamOption.setCurrentObject(conceptDriftStream4);
							conceptDriftStream6.driftstreamOption.setCurrentObject(s7);
							conceptDriftStream6.positionOption.setValue(30000);
							conceptDriftStream6.widthOption.setValue(1000);
							conceptDriftStream6.prepareForUse();
							trueChanges.add(30000.0);

							conceptDriftStream = new ConceptDriftStream();
							conceptDriftStream.streamOption.setCurrentObject(conceptDriftStream6);
							conceptDriftStream.driftstreamOption.setCurrentObject(s8);
							conceptDriftStream.positionOption.setValue(35000);
							conceptDriftStream.widthOption.setValue(1000);
							conceptDriftStream.prepareForUse();
							trueChanges.add(35000.0);
							break;
						case 5:

							break;
						case 6:

							break;
						case 7:

							break;
						case 8:
							break;
						case 9:

							break;
						case 10:
							break;
						}

						// Creating the StreamHiCS system
						adapter = createSummarisationAdapter(summarisation);
						contrastEvaluator = new Contrast(m, alpha, adapter);
						CorrelationSummary correlationSummary = new CorrelationSummary(numberOfDimensions, horizon);
						subspaceBuilder = createSubspaceBuilder(buildup, correlationSummary);
						ChangeChecker changeChecker = new TimeCountChecker(numInstances);
						streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator,
								subspaceBuilder, changeChecker, callback, correlationSummary, stopwatch);
						changeChecker.setCallback(streamHiCS);

						cscd = new CorrelatedSubspacesChangeDetector(numberOfDimensions, streamHiCS);
						cscd.prepareForUse();
						streamHiCS.setCallback(cscd);

						double[] cscdSums = new double[4];
						double[] refSums = new double[4];

						for (int i = 0; i < numberTestRuns; i++) {
							double[][] performanceMeasures = testRun(trueChanges);
							for (int j = 0; j < 4; j++) {
								cscdSums[j] += performanceMeasures[0][j];
								refSums[j] += performanceMeasures[1][j];
							}
						}

						// Calculate results
						cscdSums[4] = stopwatch.getTime("Evaluation");
						cscdSums[5] = stopwatch.getTime("Adding");
						cscdSums[6] = stopwatch.getTime("Total_CSCD");
						refSums[6] = stopwatch.getTime("Total_REF");

						for (int j = 0; j < 7; j++) {
							cscdSums[j] /= numberTestRuns;
							refSums[j] /= numberTestRuns;
						}

						String cscdMeasures = "CSCD, " + test + "," + cscdSums[0] + ", " + cscdSums[1] + ", "
								+ cscdSums[2] + ", " + cscdSums[4] + ", " + cscdSums[5] + ", " + cscdSums[6];
						String refMeasures = "REF, " + test + "," + refSums[0] + ", " + refSums[1] + ", " + refSums[2]
								+ ", " + refSums[4] + ", " + refSums[5] + ", " + refSums[6];
						System.out.println(cscdMeasures);
						System.out.println(refMeasures);

						for (int i = 0; i < 7; i++) {
							resultSummary[1][i] += cscdSums[i];
							resultSummary[2][i] += refSums[i];
						}

						results.add(cscdMeasures);
						results.add(refMeasures);
					}

					for (int i = 0; i < 7; i++) {
						resultSummary[0][i] /= 10;
						resultSummary[1][i] /= 10;
					}
					String avgCSCD = "CSCD, ";
					String avgREF = "REF, ";
					for (int i = 0; i < 6; i++) {
						avgCSCD += resultSummary[1][i] + ",";
						avgREF += resultSummary[1][i] + ",";
					}
					avgCSCD += resultSummary[0][6];
					avgREF += resultSummary[1][6];
					System.out.println(avgCSCD);
					System.out.print(avgREF);
				}
			}
		}

		// Write the results
		String filePath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/StreamHiCS/GaussianStreams/Standard/Results.txt";

		try {
			Files.write(Paths.get(filePath), results, StandardOpenOption.APPEND);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private SummarisationAdapter createSummarisationAdapter(StreamSummarisation ss) {
		boolean addDescription = false;
		if (summarisationDescription == null) {
			addDescription = true;
		}
		SummarisationAdapter adapter = null;
		switch (ss) {
		case SLIDINGWINDOW:
			aprioriThreshold = 0.2;
			hierarchicalThreshold = 0.25;
			adapter = new SlidingWindowAdapter(numberOfDimensions, horizon);
			summarisationDescription = "Sliding window, window size: " + horizon;
			break;
		case CLUSTREAM:
			aprioriThreshold = 0.3;
			hierarchicalThreshold = 0.35;
			Clustream cluStream = new Clustream();
			cluStream.kernelRadiFactorOption.setValue(2);
			int numberKernels = 400;
			cluStream.maxNumKernelsOption.setValue(numberKernels);
			cluStream.prepareForUse();
			adapter = new MicroclusteringAdapter(cluStream);
			summarisationDescription = "CluStream, maximum number kernels: " + numberKernels;
			break;
		case DENSTREAM:
			aprioriThreshold = 0.3;
			hierarchicalThreshold = 0.35;
			WithDBSCAN denStream = new WithDBSCAN();
			int speed = 100;
			double epsilon = 0.5;
			double beta = 0.2;
			double mu = 10;
			denStream.speedOption.setValue(speed);
			denStream.epsilonOption.setValue(epsilon);
			denStream.betaOption.setValue(beta);
			denStream.muOption.setValue(mu);
			// lambda calculated from horizon
			double lambda = -Math.log(0.01) / Math.log(2) / (double) horizon;
			denStream.lambdaOption.setValue(lambda);
			denStream.prepareForUse();
			adapter = new MicroclusteringAdapter(denStream);
			summarisationDescription = "DenStream, speed: " + speed + ", epsilon: " + epsilon + ", beta" + beta + ", mu"
					+ mu + ", lambda" + lambda;
			break;
		case CLUSTREE_DEPTHFIRST:
			aprioriThreshold = 0.3;
			hierarchicalThreshold = 0.35;
			ClusTree clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree, horizon: " + horizon;
			break;
		case CLUSTREE_BREADTHFIRST:
			aprioriThreshold = 0.2;
			hierarchicalThreshold = 0.25;
			clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.breadthFirstSearchOption.set();
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree, horizon: " + horizon;
			break;
		case ADAPTINGCENTROIDS:
			aprioriThreshold = 0.25;
			hierarchicalThreshold = 0.35;
			double radius = 3.5;
			double learningRate = 0.1;
			adapter = new CentroidsAdapter(horizon, radius, learningRate, "adapting");
			summarisationDescription = "Radius centroids, horizon: " + horizon + ", radius: " + radius
					+ ", learning rate: " + learningRate;
			break;
		case RADIUSCENTROIDS:
			aprioriThreshold = 0.25;
			hierarchicalThreshold = 0.35;
			radius = 3.5;
			adapter = new CentroidsAdapter(horizon, radius, 0.1, "readius");
			summarisationDescription = "Radius centroids, horizon: " + horizon + ", radius: " + radius;
			break;
		default:
			adapter = null;
		}
		if (addDescription) {
			results.add(summarisationDescription);
		}
		return adapter;
	}

	private SubspaceBuilder createSubspaceBuilder(SubspaceBuildup sb, CorrelationSummary correlationSummary) {
		cutoff = 8;
		pruningDifference = 0.15;
		boolean addDescription = false;
		if (builderDescription == null) {
			addDescription = true;
		}
		SubspaceBuilder builder = null;
		switch (sb) {
		case APRIORI:
			builder = new AprioriBuilder(numberOfDimensions, aprioriThreshold, cutoff, contrastEvaluator,
					correlationSummary);
			builderDescription = "Apriori, threshold: " + aprioriThreshold + ", cutoff: " + cutoff;
			break;
		case HIERARCHICAL:
			builder = new HierarchicalBuilderCutoff(numberOfDimensions, hierarchicalThreshold, cutoff,
					contrastEvaluator, correlationSummary, true);
			builderDescription = "Hierarchical, threshold: " + hierarchicalThreshold + ", cutoff: " + cutoff;
			break;
		default:
			builder = null;
		}
		if (addDescription) {
			results.add(builderDescription);
		}
		return builder;
	}

	private double[][] testRun(ArrayList<Double> trueChanges) {
		streamHiCS.clear();

		int numberSamples = 0;

		ArrayList<Double> cscdDetectedChanges = new ArrayList<Double>();
		ArrayList<Double> refDetectedChanges = new ArrayList<Double>();
		while (conceptDriftStream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = conceptDriftStream.nextInstance();
			stopwatch.start("Total_CSCD");
			cscd.trainOnInstance(inst);
			stopwatch.stop("Total_CSCD");
			stopwatch.start("Total_REF");
			refDetector.trainOnInstance(inst);
			stopwatch.stop("Total_REF");

			if (cscd.isWarningDetected()) {
				// System.out.println("cscd: WARNING at " + numberSamples);
			} else if (cscd.isChangeDetected()) {
				cscdDetectedChanges.add((double) numberSamples);
				System.out.println("cscd: CHANGE at " + numberSamples);
			}

			if (refDetector.isWarningDetected()) {
				// System.out.println("refDetector: WARNING at " +
				// numberSamples);
			} else if (refDetector.isChangeDetected()) {
				refDetectedChanges.add((double) numberSamples);
				System.out.println("refDetector: CHANGE at " + numberSamples);
			}

			numberSamples++;
		}

		double[] cscdPerformanceMeasures = Evaluator.evaluateConceptChange(trueChanges, cscdDetectedChanges,
				numInstances);
		double[] refPerformanceMeasures = Evaluator.evaluateConceptChange(trueChanges, refDetectedChanges,
				numInstances);
		double[][] performanceMeasures = new double[2][4];
		performanceMeasures[0] = cscdPerformanceMeasures;
		performanceMeasures[1] = refPerformanceMeasures;
		return performanceMeasures;
	}
}
