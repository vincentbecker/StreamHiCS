package clustree;

/*
 *    AuxiliaryFunctions.java
 *    Copyright (C) 2010 RWTH Aachen University, Germany
 *    @author Sanchez Villaamil (moa@cs.rwth-aachen.de)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */

public class AuxiliaryFunctions {

	public static final int GAMMA_ITERATIONS = 100;

	private AuxiliaryFunctions() {
	}

	public static void addIntegerArrays(int[] a1, int[] a2) {
		assert (a1.length == a2.length);

		for (int i = 0; i < a1.length; i++) {
			a1[i] += a2[i];
		}
	}

	public static void overwriteDoubleArray(double[] a1, double[] a2) {
		assert (a1.length == a2.length);

		for (int i = 0; i < a1.length; i++) {
			a1[i] = a2[i];
		}
	}

	public static void overwriteIntegerArray(int[] a1, int[] a2) {
		assert (a1.length == a2.length);

		for (int i = 0; i < a1.length; i++) {
			a1[i] = a2[i];
		}
	}

	public static double weight(double negLambda, long timeDifference) {
		assert (negLambda < 0);
		assert (timeDifference > 0);
		return Math.pow(2.0, negLambda * timeDifference);
	}

	public static void printArray(double[] a) {
		System.out.println(formatArray(a));
	}

	public static String formatArray(double[] a) {
		if (a.length == 0) {
			return "[]";
		}

		String res = "[";
		for (int i = 0; i < a.length - 1; i++) {
			res += Math.round(a[i] * 1000.0) / 1000.0 + ", ";
		}
		res += Math.round(a[a.length - 1] * 1000.0) / 1000.0 + "]";
		return res;
	}

	public static void printArray(String[] a) {
		if (a.length == 0) {
			System.out.println("[]");
			return;
		}

		System.out.print("[");
		for (int i = 0; i < a.length - 1; i++) {
			System.out.print(a[i] + ", ");
		}
		System.out.print(a[a.length - 1]);
		System.out.println("]");
	}

	public static double gammaHalf(int n) {
		int[] doubleFac = new int[] { 1, 1, 2, 3, 8, 15, 48, 105, 384, 945, 3840, 10395, 46080, 135135, 645120, 2027025,
				10321920, 34459425, 185794560, 654729075 };

		if (n == 0) {
			return Double.POSITIVE_INFINITY;
		}

		// Integers are simple fac(n)
		if ((n % 2) == 0) {
			int v = (n / 2) - 1;
			int res = 1;

			for (int i = 1; i <= v; i++) {
				res *= i;
			}

			return res;
		}

		// First two would yeald negative double factorials
		if (n == 1) {
			return 1.0692226492664116 / 0.6032442812094465;
		}
		if (n == 3) {
			return 0.947573901083827 / 1.0692226492664116;
		}

		return Math.sqrt(Math.PI) * (doubleFac[n - 2]) / (Math.pow(2, (n - 1) * .5));
	}

	public static double distanceProbabilty(double threshold, int dimension) {
		// threshold = (threshold*threshold) * .5;
		if (threshold == 0) {
			return 1;
		}

		// return 1 - (incompleteGamma(dimension * .5, threshold) /
		// gammaHalf(dimension));
		return 1 - (Gamma.incompleteGamma(dimension * .5, threshold * .5) / gammaHalf(dimension));
	}

	public static double gompertzWeight(double average, double count) {
		if (average < 2.0)
			return 1.0;

		double logT = Math.log(0.97);
		double logt = Math.log(0.0001);

		double denomB = (Math.pow(logT * logT, 1 / (average - 2.0)));

		double b = (Math.pow(logt * logt, 1.0 / (2.0 * (1.0 - (2.0 / average))))) / denomB;
		double c = -(1.0 / average) * Math.log(-(1.0 / b) * logT);

		assert (b >= 0) : "Bad b " + b + ", average " + average;
		assert (c >= 0) : "Bad c " + c + ", average " + average;

		// Should be okay, the following test fails for some numerica
		// bad examples
		// assert (Math.exp(-b * Math.exp(-c * average)) > 0.95);
		// assert (Math.exp(-b * Math.exp(-c * 2)) < 0.001);

		return Math.exp(-b * Math.exp(-c * count));
	}

	public static void sortDoubleArray(double[] a) {
		int i, j;
		double value;
		for (i = 1; i < a.length; i++) {
			j = i;
			value = a[j];
			while (j > 0 && a[j - 1] > value) {
				a[j] = a[j - 1];
				j--;
			}
			a[j] = value;
		}

	}

	public static double inverseError(double x) {
		double z = Math.sqrt(Math.PI) * x;
		double res = (z) / 2;

		double z2 = z * z;
		double zProd = z * z2; // z^3
		res += (1.0 / 24) * zProd;

		zProd *= z2; // z^5
		res += (7.0 / 960) * zProd;

		zProd *= z2; // z^7
		res += (127 * zProd) / 80640;

		zProd *= z2; // z^9
		res += (4369 * zProd) / 116121600;

		zProd *= z2; // z^11
		res += (34807 * zProd) / 3649536;

		zProd *= z2; // z^13
		res += (20036983 * zProd) / 797058662400d;

		/*
		 * zProd *= z2; // z^15 res += (2280356863 * zProd)/334764638208000;
		 */

		// +(49020204823 pi^(17/2) x^17)/26015994740736000+(65967241200001
		// pi^(19/2) x^19)/124564582818643968000+(15773461423793767 pi^(21/2)
		// x^21)/104634249567660933120000+O(x^22)

		return res;
	}
}