package evaluator;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Test;

import environment.Evaluator;

public class ConceptChangeMeasures {

	@Test
	public void test1() {
		ArrayList<Double> trueChanges = new ArrayList<Double>(5);
		ArrayList<Double> detectedChanges = new ArrayList<Double>(7);
		int changeLength = 1;
		int streamLength = 10;

		trueChanges.add(1.0);
		trueChanges.add(3.0);
		trueChanges.add(5.0);
		trueChanges.add(7.0);
		trueChanges.add(9.0);

		detectedChanges.add(0.5);
		detectedChanges.add(2.0);
		detectedChanges.add(2.5);
		detectedChanges.add(4.0);
		detectedChanges.add(5.5);
		// No false alarm, since within changeLength
		detectedChanges.add(5.7);
		detectedChanges.add(6.5);
		detectedChanges.add(10.0);

		double[] results = Evaluator.evaluateConceptChange(trueChanges, detectedChanges, changeLength, streamLength);
		double mtfa = results[0];
		System.out.println("MTFA: " + mtfa);
		double mtd = results[1];
		System.out.println("MTD: " + mtd);
		double mdr = results[2];
		System.out.println("MDR: " + mdr);
		double mtr = results[3];
		System.out.println("MTR: " + mtr);

		assertEquals(mtfa, 2.5, 0.01);
		assertEquals(mtd, 3.5 / 4, 0.01);
		assertEquals(mdr, 1.0 / 5, 0.01);
		assertEquals(mtr, 16.0 / 7, 0.01);
	}

	@Test
	public void test2() {
		ArrayList<Double> trueChanges = new ArrayList<Double>(1);
		ArrayList<Double> detectedChanges = new ArrayList<Double>(1);
		int changeLength = 1;
		int streamLength = 10;

		double[] results = Evaluator.evaluateConceptChange(trueChanges, detectedChanges, changeLength, streamLength);
		double mtfa = results[0];
		System.out.println("MTFA: " + mtfa);
		double mtd = results[1];
		System.out.println("MTD: " + mtd);
		double mdr = results[2];
		System.out.println("MDR: " + mdr);
		double mtr = results[3];
		System.out.println("MTR: " + mtr);

		assertEquals(mtfa, 10.0, 0.01);
		assertEquals(mtd, 0, 0.01);
		assertEquals(mdr, 1, 0.01);
		assertEquals(mtr, 0, 0.01);
	}

	@Test
	public void test3() {
		ArrayList<Double> trueChanges = new ArrayList<Double>(4);
		ArrayList<Double> detectedChanges = new ArrayList<Double>(7);
		int changeLength = 1;

		int streamLength = 10;

		trueChanges.add(1.0);
		trueChanges.add(4.0);
		trueChanges.add(6.0);
		trueChanges.add(7.0);

		detectedChanges.add(0.5);
		detectedChanges.add(2.0);
		detectedChanges.add(3.0);
		detectedChanges.add(5.0);
		detectedChanges.add(7.0);
		detectedChanges.add(8.5);
		detectedChanges.add(9.5);

		double[] results = Evaluator.evaluateConceptChange(trueChanges, detectedChanges, changeLength, streamLength);
		double mtfa = results[0];
		System.out.println("MTFA: " + mtfa);
		double mtd = results[1];
		System.out.println("MTD: " + mtd);
		double mdr = results[2];
		System.out.println("MDR: " + mdr);
		double mtr = results[3];
		System.out.println("MTR: " + mtr);

		assertEquals(mtfa, 2.5, 0.01);
		assertEquals(mtd, 1.125, 0.01);
		assertEquals(mdr, 0.0, 0.01);
		assertEquals(mtr, 20.0 / 9, 0.01);
	}
	
	@Test
	public void test4() {
		ArrayList<Double> trueChanges = new ArrayList<Double>(3);
		ArrayList<Double> detectedChanges = new ArrayList<Double>(0);
		int changeLength = 1000;

		int streamLength = 20000;

		trueChanges.add(4500.0);
		trueChanges.add(9500.0);
		trueChanges.add(14500.0);

		detectedChanges.add(6337.0);
		detectedChanges.add(9964.0);
		detectedChanges.add(10028.0);
		detectedChanges.add(10104.0);
		detectedChanges.add(10382.0);
		detectedChanges.add(13943.0);
		detectedChanges.add(15555.0);
		detectedChanges.add(15877.0);


		double[] results = Evaluator.evaluateConceptChange(trueChanges, detectedChanges, changeLength, streamLength);
		double mtfa = results[0];
		System.out.println("MTFA: " + mtfa);
		double mtd = results[1];
		System.out.println("MTD: " + mtd);
		double mdr = results[2];
		System.out.println("MDR: " + mdr);
		double mtr = results[3];
		System.out.println("MTR: " + mtr);

		assertEquals(mtfa, 6666.6667, 0.01);
		assertEquals(mtd, 3356.0/3, 0.01);
		assertEquals(mdr, 0.0, 0.01);
		assertEquals(mtr, 6666.6667 * 3 / 3356, 0.01);
	}
}
