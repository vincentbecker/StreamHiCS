package streamtests;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Before;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import contrast.Contrast;
import contrast.MicroclusterContrast;
import environment.CSVReader;
import environment.CovarianceMatrixGenerator;
import environment.Evaluator;
import fullsystem.Callback;
import fullsystem.StreamHiCS;
import moa.clusterers.clustree.ClusTree;
import moa.streams.ConceptDriftStream;
import streams.GaussianStream;
import streams.UncorrelatedStream;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class HighDimensionalConceptDriftTest {

	private static final String path = "Tests/CovarianceMatrices/";
	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			// System.out.println("onAlarm().");
		}
	};
	private ConceptDriftStream conceptDriftStream;
	private StreamHiCS streamHiCS;
	private ArrayList<SubspaceSet> correctResults;
	private int testCounter = 0;
	private double scoreSum = 0;
	private final int numInstances = 35000;
	private int numberSamples = 0;
	private int numberOfDimensions = 50;
	private int blockSize = 10;
	
	@Before
	public void setUp() throws Exception {
		numberSamples = 0;
	}

	@Test
	public void test1() {
		correctResults = new ArrayList<SubspaceSet>();
		UncorrelatedStream s1 = new UncorrelatedStream();
		s1.dimensionsOption.setValue(50);
		s1.scaleOption.setValue(10);
		s1.prepareForUse();

		double[][] covarianceMatrix2 = CovarianceMatrixGenerator.generateCovarianceMatrix(numberOfDimensions, 0,
				0.9);
		GaussianStream s2 = new GaussianStream(covarianceMatrix2);

		double[][] covarianceMatrix3 = CovarianceMatrixGenerator.generateCovarianceMatrix(numberOfDimensions, 0,
				0.9);
		GaussianStream s3 = new GaussianStream(covarianceMatrix3);

		double[][] covarianceMatrix4 = CovarianceMatrixGenerator.generateCovarianceMatrix(numberOfDimensions, 0,
				0.9);
		GaussianStream s4 = new GaussianStream(covarianceMatrix4);

		ConceptDriftStream cds1 = new ConceptDriftStream();
		cds1.streamOption.setCurrentObject(s1);
		cds1.driftstreamOption.setCurrentObject(s2);
		cds1.positionOption.setValue(10000);
		cds1.widthOption.setValue(500);
		cds1.prepareForUse();

		ConceptDriftStream cds2 = new ConceptDriftStream();
		cds2.streamOption.setCurrentObject(cds1);
		cds2.driftstreamOption.setCurrentObject(s3);
		cds2.positionOption.setValue(20000);
		cds2.widthOption.setValue(500);
		cds2.prepareForUse();

		ConceptDriftStream conceptDriftStream = new ConceptDriftStream();
		conceptDriftStream.streamOption.setCurrentObject(cds2);
		conceptDriftStream.driftstreamOption.setCurrentObject(s4);
		conceptDriftStreams.positionOption.setValue(30000);
		conceptDriftStream.widthOption.setValue(500);
		conceptDriftStream.prepareForUse();

		// Adding the expected results for evaluation
		SubspaceSet cr5000 = new SubspaceSet();
		correctResults.add(cr5000);

		SubspaceSet cr10000 = new SubspaceSet();
		correctResults.add(cr10000);

		SubspaceSet cr15000 = new SubspaceSet();
		correctResults.add(cr15000);

		SubspaceSet cr20000 = new SubspaceSet();
		correctResults.add(cr20000);

		SubspaceSet cr25000 = new SubspaceSet();
		cr25000.addSubspace(new Subspace(0, 1, 2));
		correctResults.add(cr25000);

		SubspaceSet cr30000 = new SubspaceSet();
		cr30000.addSubspace(new Subspace(0, 1, 2));
		correctResults.add(cr30000);

		SubspaceSet cr35000 = new SubspaceSet();
		cr35000.addSubspace(new Subspace(0, 1));
		cr35000.addSubspace(new Subspace(2, 3, 4));
		correctResults.add(cr35000);
		
		ClusTree mcs = new ClusTree();
		mcs.horizonOption.setValue(4000);
		mcs.resetLearningImpl();

		double alpha = 0.15;
		double epsilon = 0;
		double threshold = 0.35;
		int cutoff = 8;
		double pruningDifference = 0.2;

		Contrast contrastEvaluator = new MicroclusterContrast(50, alpha, mcs);
		ChangeChecker changeChecker = new TimeCountChecker(5000);
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(5, threshold, cutoff, pruningDifference,
				contrastEvaluator);
		//SubspaceBuilder subspaceBuilder = new FastBuilder(5, threshold, pruningDifference, contrastEvaluator);
		this.streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder, changeChecker,
				callback);
		changeChecker.setCallback(streamHiCS);

		while (conceptDriftStream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = conceptDriftStream.nextInstance();
			streamHiCS.add(inst);
			numberSamples++;
			if (numberSamples != 0 && numberSamples % 5000 == 0) {
				evaluate();
			}
		}

		double meanScore = scoreSum / testCounter;
		System.out.println("Mean score: " + meanScore);
		assertTrue(meanScore >= 0.75);
	}

	private void evaluate() {
		System.out.println("Number of samples: " + numberSamples);
		SubspaceSet correctResult = correctResults.get(testCounter);
		scoreSum += Evaluator.evaluateTPvsFP(streamHiCS.getCurrentlyCorrelatedSubspaces(), correctResult);
		testCounter++;
	}
}
