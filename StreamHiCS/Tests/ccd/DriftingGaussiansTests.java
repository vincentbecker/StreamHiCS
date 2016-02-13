package ccd;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import clustree.ClusTree;
import environment.AccuracyEvaluator;
import environment.Evaluator;
import environment.Stopwatch;
import environment.Parameters.StreamSummarisation;
import environment.Parameters.SubspaceBuildup;
import fullsystem.Contrast;
import fullsystem.SubspaceChangeDetectors;
import fullsystem.SubspaceClassifiersChangeDetector;
import fullsystem.FullSpaceChangeDetector;
import fullsystem.StreamHiCS;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.trees.HoeffdingTree;
import streamdatastructures.CentroidsAdapter;
import streamdatastructures.CorrelationSummary;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SummarisationAdapter;
import streams.DriftingGaussians;
import streams.SubspaceRandomRBFGeneratorDrift;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.CliqueBuilder;
import subspacebuilder.ComponentBuilder;
import subspacebuilder.HierarchicalBuilderCutoff;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;
import weka.core.Utils;

public class DriftingGaussiansTests {
	private SubspaceRandomRBFGeneratorDrift stream;
	private int numInstances;
	private final int horizon = 2000;
	private final int m = 50;
	private final double alpha = 0.05;
	private double epsilon;
	private double aprioriThreshold;
	private double hierarchicalThreshold;
	private double connectedComponentsThreshold;
	private double cliqueThreshold;
	private int cutoff;
	private double pruningDifference;
	private int numberOfDimensions;
	private static Stopwatch stopwatch;
	private static final int numberTestRuns = 10;
	private List<String> results;
	private String summarisationDescription = null;
	private String builderDescription = null;
	private SubspaceChangeDetectors scd;
	private SubspaceClassifiersChangeDetector sccd;
	private FullSpaceChangeDetector refDetector;
	private Random random;

	@BeforeClass
	public static void setUpBeforeClass() {
		stopwatch = new Stopwatch();
	}

	@Before
	public void before() {
		random = new Random();
	}

	@Test
	public void test() {
		// Output
		results = new LinkedList<String>();
		for (StreamSummarisation summarisation : StreamSummarisation.values()) {
			summarisationDescription = null;
			for (SubspaceBuildup buildup : SubspaceBuildup.values()) {
				builderDescription = null;
				if (summarisation == StreamSummarisation.ADAPTINGCENTROIDS
						&& buildup == SubspaceBuildup.CONNECTED_COMPONENTS) {
					for (int test = 4; test <= 4; test++) {
						stopwatch.reset();
						double threshold = 1;

						ArrayList<Double> trueChanges = new ArrayList<Double>();
						int[] changePoints = null;
						int changeLength = 0;
						double speed = 0;
						int[] virtualDriftPoints = null;
						results.add("" + test);
						switch (test) {
						case 1:
							switch (summarisation) {
							case CLUSTREE_DEPTHFIRST:
								aprioriThreshold = 0.2;
								hierarchicalThreshold = 0.5;
								break;
							case ADAPTINGCENTROIDS:
								aprioriThreshold = 0.25;
								hierarchicalThreshold = 0.65;
								break;
							default:
								break;
							}
							switch (buildup) {
							case APRIORI:
								threshold = aprioriThreshold;
								connectedComponentsThreshold = 0.5;
								break;
							case HIERARCHICAL:
								threshold = hierarchicalThreshold;
								connectedComponentsThreshold = 0.5;
								break;
							case CONNECTED_COMPONENTS:
								threshold = connectedComponentsThreshold;
								connectedComponentsThreshold = 0.5;
								break;
							default:
								break;
							}
							epsilon = 0.15;
							numberOfDimensions = 6;
							numInstances = 25000;
							stream = new SubspaceRandomRBFGeneratorDrift();
							stream.numAttsOption.setValue(numberOfDimensions);
							stream.numClassesOption.setValue(2);
							stream.avgSubspaceSizeOption.setValue(numberOfDimensions / 2);
							stream.scaleIrrelevantDimensionsOption.setValue(4);
							stream.numCentroidsOption.setValue(10);
							stream.numSubspaceCentroidsOption.setValue(10);
							stream.sameSubspaceOption.setValue(true);
							stream.randomSubspaceSizeOption.setValue(false);
							stream.numDriftCentroidsOption.setValue(10);
							changePoints = new int[4];
							changePoints[0] = 5000;
							changePoints[1] = 10000;
							changePoints[2] = 15000;
							changePoints[3] = 20000;
							changeLength = 1000;
							speed = 0.5;
							trueChanges.add(5000.0);
							trueChanges.add(10000.0);
							trueChanges.add(15000.0);
							trueChanges.add(20000.0);
							break;
						case 2:
							switch (summarisation) {
							case CLUSTREE_DEPTHFIRST:
								aprioriThreshold = 0.3;
								hierarchicalThreshold = 0.25;
								connectedComponentsThreshold = 0.5;
								break;
							case ADAPTINGCENTROIDS:
								aprioriThreshold = 0.25;
								hierarchicalThreshold = 0.3;
								connectedComponentsThreshold = 0.3;
								break;
							default:
								break;
							}
							switch (buildup) {
							case APRIORI:
								threshold = aprioriThreshold;
								break;
							case HIERARCHICAL:
								threshold = hierarchicalThreshold;
								break;
							case CONNECTED_COMPONENTS:
								threshold = connectedComponentsThreshold;
								break;
							default:
								break;
							}
							numberOfDimensions = 10;
							epsilon = 0.15;
							numInstances = 25000;
							stream = new SubspaceRandomRBFGeneratorDrift();
							stream.numAttsOption.setValue(numberOfDimensions);
							stream.numClassesOption.setValue(2);
							stream.avgSubspaceSizeOption.setValue(numberOfDimensions / 2);
							stream.numCentroidsOption.setValue(5);
							stream.scaleIrrelevantDimensionsOption.setValue(4);
							stream.numSubspaceCentroidsOption.setValue(5);
							stream.sameSubspaceOption.setValue(true);
							stream.randomSubspaceSizeOption.setValue(false);
							stream.numDriftCentroidsOption.setValue(5);
							changePoints = new int[4];
							changePoints[0] = 5000;
							changePoints[1] = 10000;
							changePoints[2] = 15000;
							changePoints[3] = 20000;
							changeLength = 500;
							speed = 0.5;
							trueChanges.add(5000.0);
							trueChanges.add(10000.0);
							trueChanges.add(15000.0);
							trueChanges.add(20000.0);
							break;

						case 3:
							switch (summarisation) {
							case CLUSTREE_DEPTHFIRST:
								aprioriThreshold = 0.35;
								hierarchicalThreshold = 0.5;
								connectedComponentsThreshold = 0.5;
								cliqueThreshold = 0.55;
								break;
							case ADAPTINGCENTROIDS:
								aprioriThreshold = 0.3;
								hierarchicalThreshold = 0.65;
								connectedComponentsThreshold = 0.35;
								cliqueThreshold = 0.4;
								break;
							default:
								break;
							}
							switch (buildup) {
							case APRIORI:
								threshold = aprioriThreshold;
								break;
							case HIERARCHICAL:
								threshold = hierarchicalThreshold;
								break;
							case CONNECTED_COMPONENTS:
								threshold = connectedComponentsThreshold;
								break;
							case CLIQUE:
								threshold = cliqueThreshold;
								break;
							}
							numberOfDimensions = 50;
							numInstances = 90000;
							epsilon = 0.1;
							stream = new DriftingGaussians();
							stream.numAttsOption.setValue(numberOfDimensions);
							stream.numClassesOption.setValue(2);
							stream.avgSubspaceSizeOption.setValue(5);
							stream.numCentroidsOption.setValue(2);
							stream.numSubspaceCentroidsOption.setValue(2);
							stream.sameSubspaceOption.setValue(true);
							stream.randomSubspaceSizeOption.setValue(false);
							stream.numDriftCentroidsOption.setValue(2);
							changePoints = new int[4];
							changePoints[0] = 10000;
							changePoints[1] = 30000;
							changePoints[2] = 50000;
							changePoints[3] = 70000;

							changeLength = 1000;
							speed = 5;
							trueChanges.add(10000.0);
							trueChanges.add(30000.0);
							trueChanges.add(50000.0);
							trueChanges.add(70000.0);

							virtualDriftPoints = null;
							break;
						case 4:
							switch (summarisation) {
							case CLUSTREE_DEPTHFIRST:
								aprioriThreshold = 0.35;
								hierarchicalThreshold = 0.5;
								connectedComponentsThreshold = 0.45;
								cliqueThreshold = 0.55;
								break;
							case ADAPTINGCENTROIDS:
								aprioriThreshold = 0.3;
								hierarchicalThreshold = 0.65;
								connectedComponentsThreshold = 0.35;
								cliqueThreshold = 0.5;
								break;
							default:
								break;
							}
							switch (buildup) {
							case APRIORI:
								threshold = aprioriThreshold;
								break;
							case HIERARCHICAL:
								threshold = hierarchicalThreshold;
								break;
							case CONNECTED_COMPONENTS:
								threshold = connectedComponentsThreshold;
								break;
							case CLIQUE:
								threshold = cliqueThreshold;
								break;
							}
							numberOfDimensions = 50;
							numInstances = 90000;
							epsilon = 0.1;
							stream = new DriftingGaussians();
							stream.numAttsOption.setValue(numberOfDimensions);
							stream.numClassesOption.setValue(2);
							stream.avgSubspaceSizeOption.setValue(numberOfDimensions / 10);
							stream.numCentroidsOption.setValue(2);
							stream.sameSubspaceOption.setValue(true);
							stream.randomSubspaceSizeOption.setValue(false);
							stream.numDriftCentroidsOption.setValue(2);
							changePoints = new int[4];
							changePoints[0] = 10000;
							changePoints[1] = 30000;
							changePoints[2] = 50000;
							changePoints[3] = 70000;

							changeLength = 1000;
							speed = 5;
							trueChanges.add(10000.0);
							trueChanges.add(30000.0);
							trueChanges.add(50000.0);
							trueChanges.add(70000.0);

							virtualDriftPoints = new int[4];
							virtualDriftPoints[0] = 20000;
							virtualDriftPoints[1] = 40000;
							virtualDriftPoints[2] = 60000;
							virtualDriftPoints[3] = 80000;
							break;
						}

						// Creating the SCD system
						SummarisationAdapter adapter1 = createSummarisationAdapter(summarisation);
						Contrast contrastEvaluator1 = new Contrast(m, alpha, adapter1);
						CorrelationSummary correlationSummary1 = new CorrelationSummary(numberOfDimensions, horizon);
						SubspaceBuilder subspaceBuilder1 = createSubspaceBuilder(buildup, contrastEvaluator1,
								correlationSummary1);
						ChangeChecker changeChecker1 = new TimeCountChecker(1000);
						StreamHiCS streamHiCS1 = new StreamHiCS(epsilon, threshold, pruningDifference,
								contrastEvaluator1, subspaceBuilder1, changeChecker1, null, correlationSummary1,
								stopwatch);
						changeChecker1.setCallback(streamHiCS1);
						scd = new SubspaceChangeDetectors(numberOfDimensions, streamHiCS1);
						scd.useRestspaceOption.setValue(false);
						scd.prepareForUse();
						streamHiCS1.setCallback(scd);

						// Creating the SCCD system
						SummarisationAdapter adapter2 = createSummarisationAdapter(summarisation);
						Contrast contrastEvaluator2 = new Contrast(m, alpha, adapter2);
						CorrelationSummary correlationSummary2 = new CorrelationSummary(numberOfDimensions, horizon);
						SubspaceBuilder subspaceBuilder2 = createSubspaceBuilder(buildup, contrastEvaluator2,
								correlationSummary2);
						ChangeChecker changeChecker2 = new TimeCountChecker(1000);
						StreamHiCS streamHiCS2 = new StreamHiCS(epsilon, threshold, pruningDifference,
								contrastEvaluator2, subspaceBuilder2, changeChecker2, null, correlationSummary2,
								stopwatch);
						changeChecker2.setCallback(streamHiCS2);
						sccd = new SubspaceClassifiersChangeDetector(numberOfDimensions, streamHiCS2);
						sccd.useRestspaceOption.setValue(false);
						sccd.prepareForUse();
						streamHiCS2.setCallback(sccd);

						// Creating the reference detector
						refDetector = new FullSpaceChangeDetector();
						AbstractClassifier baseLearner = new HoeffdingTree();
						baseLearner.prepareForUse();
						refDetector.baseLearnerOption.setCurrentObject(baseLearner);
						refDetector.prepareForUse();

						double[] scdSums = new double[8];
						double[] sccdSums = new double[8];
						double[] refSums = new double[8];

						stopwatch.reset();
						for (int i = 0; i < numberTestRuns; i++) {
							// Reset the stream and the change detectors
							int seed = random.nextInt();
							stream.modelRandomSeedOption.setValue(seed);
							System.out.println("Seed1: " + seed);
							seed = random.nextInt();
							stream.instanceRandomSeedOption.setValue(seed);
							System.out.println("Seed2: " + seed);
							// stream.restart();
							stream.prepareForUse();
							scd.resetLearning();
							sccd.resetLearning();
							refDetector.resetLearning();

							System.out.println("Run: " + (i + 1));
							double[][] performanceMeasures = testRun(changePoints, changeLength, speed, trueChanges,
									virtualDriftPoints);
							for (int j = 0; j < 5; j++) {
								scdSums[j] += performanceMeasures[0][j];
								sccdSums[j] += performanceMeasures[1][j];
								refSums[j] += performanceMeasures[2][j];
							}
						}

						// Calculate results
						scdSums[5] = stopwatch.getTime("Evaluation");
						scdSums[6] = stopwatch.getTime("Adding");
						scdSums[7] = stopwatch.getTime("Total_SCD");
						sccdSums[7] = stopwatch.getTime("Total_SCCD");
						refSums[7] = stopwatch.getTime("Total_REF");

						for (int j = 0; j < 8; j++) {
							scdSums[j] /= numberTestRuns;
							sccdSums[j] /= numberTestRuns;
							refSums[j] /= numberTestRuns;
						}

						System.out.println(
								"Test, MTFA, MTD, MDR, MTR, Error rate, Evaluation time, Adding time, Total time");
						String scdMeasures = "SCD," + test + "," + scdSums[0] + ", " + scdSums[1] + ", " + scdSums[2]
								+ ", " + scdSums[3] + ", " + scdSums[4] + ", " + scdSums[5] + ", " + scdSums[6] + ", "
								+ scdSums[7];
						String sccdMeasures = "SCCD," + test + "," + sccdSums[0] + ", " + sccdSums[1] + ", "
								+ sccdSums[2] + ", " + sccdSums[3] + ", " + sccdSums[4] + ", " + sccdSums[7];
						String refMeasures = "REF," + test + "," + refSums[0] + ", " + refSums[1] + ", " + refSums[2]
								+ ", " + refSums[3] + ", " + refSums[4] + ", " + refSums[7];
						System.out.println(scdMeasures);
						System.out.println(sccdMeasures);
						System.out.println(refMeasures);
						results.add(scdMeasures);
						results.add(sccdMeasures);
						results.add(refMeasures);
					}
				}
			}
		}

		// Write the results
		String filePath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/ConceptChangeDetection/DriftingGaussians/Tests.txt";

		try {
			Files.write(Paths.get(filePath), results, StandardOpenOption.APPEND);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private double[][] testRun(int[] changePoints, int changeLength, double speed, ArrayList<Double> trueChanges,
			int[] virtualDriftPoints) {
		int numberSamples = 0;

		ArrayList<Double> scdDetectedChanges = new ArrayList<Double>();
		ArrayList<Double> sccdDetectedChanges = new ArrayList<Double>();
		ArrayList<Double> refDetectedChanges = new ArrayList<Double>();
		AccuracyEvaluator scdAccuracy = new AccuracyEvaluator();
		AccuracyEvaluator sccdAccuracy = new AccuracyEvaluator();
		AccuracyEvaluator refAccuracy = new AccuracyEvaluator();

		int changeCounter = 0;
		int driftStart = 0;
		boolean finishedDrifts = false;
		if (changePoints != null) {
			driftStart = changePoints[0];
		}

		int virtualDriftCounter = 0;
		boolean finishedVirtualDrifts = false;
		if (virtualDriftPoints != null) {
			driftStart = changePoints[0];
		}

		// cscd.onAlarm();
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			if (numberSamples % 1000 == 0) {
				// System.out.println(streamHiCS.getNumberOfElements());
				// System.out.println("Found: " +
				// cscd.getCurrentlyCorrelatedSubspaces().toString());
			}

			// Carry out the concept drift at the relevant points
			if (changePoints != null && !finishedDrifts) {
				if (numberSamples == driftStart) {
					// Before 0.05
					// 5: best 0.5
					stream.speedChangeOption.setValue(speed);
					changeCounter++;
				} else if (numberSamples > driftStart + changeLength - 1) {
					stream.speedChangeOption.setValue(0.0);
					if (changeCounter < changePoints.length) {
						driftStart = changePoints[changeCounter];
					} else {
						finishedDrifts = true;
					}

				}
			}

			if (virtualDriftPoints != null && !finishedVirtualDrifts) {
				if (numberSamples == virtualDriftPoints[virtualDriftCounter]) {
					stream.subspaceChange(changeLength);
					virtualDriftCounter++;
					if (changeCounter >= changePoints.length - 1) {
						finishedVirtualDrifts = true;
					}
				}
			}

			Instance inst = stream.nextInstance();

			// For accuracy
			int trueClass = (int) inst.classValue();
			int prediction = scd.getClassPrediction(inst);
			scdAccuracy.addClassLabel(trueClass);
			scdAccuracy.addPrediction(prediction);
			prediction = sccd.getClassPrediction(inst);
			sccdAccuracy.addClassLabel(trueClass);
			sccdAccuracy.addPrediction(prediction);
			prediction = Utils.maxIndex(refDetector.getVotesForInstance(inst));
			refAccuracy.addClassLabel(trueClass);
			refAccuracy.addPrediction(prediction);

			stopwatch.start("Total_SCD");
			scd.trainOnInstance(inst);
			stopwatch.stop("Total_SCD");
			stopwatch.start("Total_SCCD");
			sccd.trainOnInstance(inst);
			stopwatch.stop("Total_SCCD");
			stopwatch.start("Total_REF");
			refDetector.trainOnInstance(inst);
			stopwatch.stop("Total_REF");

			if (scd.isWarningDetected()) {
				// System.out.println("cscd: WARNING at " + numberSamples);
			} else if (scd.isChangeDetected()) {
				scdDetectedChanges.add((double) numberSamples);
				System.out.println("scd: CHANGE at " + numberSamples);
			}

			if (sccd.isWarningDetected()) {
				// System.out.println("cscd: WARNING at " + numberSamples);
			} else if (sccd.isChangeDetected()) {
				sccdDetectedChanges.add((double) numberSamples);
				System.out.println("sccd: CHANGE at " + numberSamples);
			}

			if (refDetector.isWarningDetected()) {
				// System.out.println("refDetector: WARNING at " +
				// numberSamples);
			} else if (refDetector.isChangeDetected()) {
				refDetectedChanges.add((double) numberSamples);
				// System.out.println("refDetector: CHANGE at " +
				// numberSamples);
			}

			numberSamples++;
		}

		double[] scdPerformanceMeasures = Evaluator.evaluateConceptChange(trueChanges, scdDetectedChanges, changeLength,
				numInstances);
		double[] sccdPerformanceMeasures = Evaluator.evaluateConceptChange(trueChanges, sccdDetectedChanges,
				changeLength, numInstances);
		double[] refPerformanceMeasures = Evaluator.evaluateConceptChange(trueChanges, refDetectedChanges, changeLength,
				numInstances);
		double[][] performanceMeasures = new double[3][5];
		for (int i = 0; i < 4; i++) {
			performanceMeasures[0][i] = scdPerformanceMeasures[i];
			performanceMeasures[1][i] = sccdPerformanceMeasures[i];
			performanceMeasures[2][i] = refPerformanceMeasures[i];
		}
		performanceMeasures[0][4] = scdAccuracy.calculateOverallErrorRate();
		performanceMeasures[1][4] = sccdAccuracy.calculateOverallErrorRate();
		performanceMeasures[2][4] = refAccuracy.calculateOverallErrorRate();
		String scdP = "SCD, ";
		String sccdP = "SCCD, ";
		String refP = "REF, ";
		for (int i = 0; i < 5; i++) {
			scdP += performanceMeasures[0][i] + ", ";
			sccdP += performanceMeasures[1][i] + ", ";
			refP += performanceMeasures[2][i] + ", ";
		}
		System.out.println(scdP);
		System.out.println(sccdP);
		System.out.println(refP);
		results.add(scdP);
		results.add(sccdP);
		results.add(refP);

		List<String> errorRatesList = new ArrayList<String>();
		double[] scdSmoothedErrorRates = scdAccuracy.calculateSmoothedErrorRates(1000);
		double[] sccdSmoothedErrorRates = sccdAccuracy.calculateSmoothedErrorRates(1000);
		double[] refSmoothedErrorRates = refAccuracy.calculateSmoothedErrorRates(1000);
		for (int i = 0; i < scdAccuracy.size(); i++) {
			errorRatesList.add(i + "," + scdSmoothedErrorRates[i] + "," + sccdSmoothedErrorRates[i] + ","
					+ refSmoothedErrorRates[i]);
		}

		String filePath = "C:/Users/Vincent/Desktop/ErrorRates_DriftingGaussians.csv";

		try {
			Files.write(Paths.get(filePath), errorRatesList);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return performanceMeasures;
	}

	private SummarisationAdapter createSummarisationAdapter(StreamSummarisation ss) {
		boolean addDescription = false;
		if (summarisationDescription == null) {
			addDescription = true;
		}
		SummarisationAdapter adapter = null;
		switch (ss) {
		case CLUSTREE_DEPTHFIRST:
			ClusTree clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree, horizon: " + horizon;
			break;
		case ADAPTINGCENTROIDS:
			double radius = 10 * Math.sqrt(numberOfDimensions) - 1;
			double learningRate = 1;
			adapter = new CentroidsAdapter(horizon, radius, learningRate, "adapting");
			summarisationDescription = "Adapting centroids, horizon: " + horizon + ", radius: " + radius
					+ ", learning rate: " + learningRate;
			break;
		default:
			adapter = null;
		}
		if (addDescription) {
			results.add(summarisationDescription);
		}
		return adapter;
	}

	private SubspaceBuilder createSubspaceBuilder(SubspaceBuildup sb, Contrast contrastEvaluator,
			CorrelationSummary correlationSummary) {
		cutoff = 3;
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
		case CONNECTED_COMPONENTS:
			pruningDifference = 0.15;
			builder = new ComponentBuilder(numberOfDimensions, connectedComponentsThreshold, contrastEvaluator,
					correlationSummary);
			builderDescription = "Connnected Components, threshold: " + connectedComponentsThreshold;
			break;
		case CLIQUE:
			pruningDifference = 0.15;
			builder = new CliqueBuilder(numberOfDimensions, cliqueThreshold, contrastEvaluator, correlationSummary);
			builderDescription = "CliqueBuilder, threshold: " + cliqueThreshold;
			break;
		default:
			builder = null;
		}
		if (addDescription) {
			results.add(builderDescription);
		}
		return builder;
	}
}
