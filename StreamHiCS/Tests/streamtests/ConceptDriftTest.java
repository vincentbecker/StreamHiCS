package streamtests;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Before;
import org.junit.Test;

import changechecker.TimeCountChecker;
import contrast.Callback;
import contrast.Contrast;
import contrast.MicroclusterContrast;
import environment.CSVReader;
import environment.Evaluator;
import fullsystem.StreamHiCS;
import moa.clusterers.clustree.ClusTree;
import moa.streams.ConceptDriftStream;
import streams.GaussianStream;
import streams.UncorrelatedStream;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import weka.core.Instance;

public class ConceptDriftTest {

	private static CSVReader csvReader;
	private static final String path = "Tests/CovarianceMatrices/";
	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			// Evaluate
			SubspaceSet correctResult = correctResults.get(testCounter - 1);
			scoreSum += Evaluator.evaluateTPvsFP(streamHiCS.getCurrentlyCorrelatedSubspaces(), correctResult);
			testCounter++;
		}
	};
	private ConceptDriftStream conceptDriftStream;
	private StreamHiCS streamHiCS;
	private ArrayList<SubspaceSet> correctResults;
	private int testCounter = 1;
	private int scoreSum = 0;
	private final int numInstances = 10000;
	private int numberSamples = 0;

	@Before
	public void setUp() throws Exception {
		csvReader = new CSVReader();
		numberSamples = 0;
	}

	@Test
	public void test1() {
		correctResults = new ArrayList<SubspaceSet>();
		UncorrelatedStream s1 = new UncorrelatedStream();
		s1.dimensionsOption.setValue(5);
		s1.scaleOption.setValue(10);
		s1.prepareForUse();

		GaussianStream s2 = new GaussianStream(csvReader.read(path + "Test1.csv"));

		GaussianStream s3 = new GaussianStream(csvReader.read(path + "Test2.csv"));

		GaussianStream s4 = new GaussianStream(csvReader.read(path + "Test3.csv"));

		GaussianStream s5 = new GaussianStream(csvReader.read(path + "Test4.csv"));

		ConceptDriftStream cds1 = new ConceptDriftStream();
		cds1.streamOption.setCurrentObject(s1);
		cds1.driftstreamOption.setCurrentObject(s2);
		cds1.positionOption.setValue(1000);
		cds1.widthOption.setValue(500);
		cds1.prepareForUse();

		ConceptDriftStream cds2 = new ConceptDriftStream();
		cds2.streamOption.setCurrentObject(cds1);
		cds2.driftstreamOption.setCurrentObject(s3);
		cds2.positionOption.setValue(3000);
		cds2.widthOption.setValue(500);
		cds2.prepareForUse();

		ConceptDriftStream cds3 = new ConceptDriftStream();
		cds3.streamOption.setCurrentObject(cds2);
		cds3.driftstreamOption.setCurrentObject(s4);
		cds3.positionOption.setValue(5000);
		cds3.widthOption.setValue(500);
		cds3.prepareForUse();

		conceptDriftStream = new ConceptDriftStream();
		conceptDriftStream.streamOption.setCurrentObject(cds3);
		conceptDriftStream.driftstreamOption.setCurrentObject(s5);
		conceptDriftStream.positionOption.setValue(7000);
		conceptDriftStream.widthOption.setValue(500);
		conceptDriftStream.prepareForUse();

		SubspaceSet cr1 = new SubspaceSet();
		correctResults.add(cr1);

		SubspaceSet cr2 = new SubspaceSet();
		cr2.addSubspace(new Subspace(0, 1));
		cr2.addSubspace(new Subspace(2, 3, 4));
		correctResults.add(cr2);

		SubspaceSet cr3 = new SubspaceSet();
		cr3.addSubspace(new Subspace(0, 1, 2, 3, 4));
		correctResults.add(cr3);

		SubspaceSet cr4 = new SubspaceSet();
		cr4.addSubspace(new Subspace(0, 1, 2));
		cr4.addSubspace(new Subspace(2, 3, 4));
		correctResults.add(cr4);

		SubspaceSet cr5 = new SubspaceSet();
		cr5.addSubspace(new Subspace(0, 1, 2));
		correctResults.add(cr5);

		ClusTree mcs = new ClusTree();
		mcs.resetLearningImpl();

		double alpha = 0.1;
		double epsilon = 0;
		double threshold = 0.25;
		int cutoff = 8;
		double pruningDifference = 0.15;

		Contrast contrastEvaluator = new MicroclusterContrast(callback, 20, alpha, mcs, new TimeCountChecker(1000));
		this.streamHiCS = new StreamHiCS(epsilon, threshold, contrastEvaluator,
				new AprioriBuilder(5, threshold, cutoff, pruningDifference, contrastEvaluator), callback);

		while (conceptDriftStream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = conceptDriftStream.nextInstance();
			streamHiCS.add(inst);
			numberSamples++;
		}

		assertTrue(scoreSum / testCounter >= 0.75);
	}

}
