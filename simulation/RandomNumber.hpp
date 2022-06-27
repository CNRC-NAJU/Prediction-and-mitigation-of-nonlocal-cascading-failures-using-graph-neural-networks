#ifndef CpT_RANDOM_NUMBER_HPP_
#define CpT_RANDOM_NUMBER_HPP_

#include <random>
#include <cstdint>
#include <limits>
#include "pcg_random.hpp"

namespace Snu {
namespace Cnrc {

	class XORShift32 {
		public:
			typedef uint32_t result_type;

		public:
			XORShift32(uint32_t val = defaultSeed) {
				seed(val);
			}

			result_type operator()() {
				uint32_t t = x ^ (x << 11);
				x = y; y = z; z = w;
				return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
			}

			static constexpr result_type min() {
				return std::numeric_limits<uint32_t>::min();
			}

			static constexpr result_type max() {
				return std::numeric_limits<uint32_t>::max();
			}

			void seed(uint32_t val = defaultSeed) {
				x = val;
				y = 362436069;
				z = 521288629;
				w = 88675123;
			}

		private:
			static const uint32_t defaultSeed = 13579;

		private:
			uint32_t x;
			uint32_t y;
			uint32_t z;
			uint32_t w;
	};

	/*
	 * TODO: Native 64-bit algorithm?
	 */
	class XORShift64 {
		public:
			typedef uint64_t result_type;

		public:
			XORShift64(uint64_t val = defaultSeed)
			: x1(static_cast<uint32_t>(val >> 32))
			, x2(static_cast<uint32_t>(val)) {
			}

			result_type operator()() {
				return ((uint64_t) x1()) << 32 | x2();
			}

			static constexpr result_type min() {
				return std::numeric_limits<uint64_t>::min();
			}

			static constexpr result_type max() {
				return std::numeric_limits<uint64_t>::max();
			}

			void seed(uint64_t val = defaultSeed) {
				x1.seed(static_cast<uint32_t>(val >> 32));
				x2.seed(static_cast<uint32_t>(val));
			}

		private:
			static const uint64_t defaultSeed = 135790135790L;

		private:
			XORShift32 x1;
			XORShift32 x2;
	};

	//typedef std::mt19937& RandomNumberEngine32;
	//typedef std::mt19937_64& RandomNumberEngine64;
	//typedef XORShift32& RandomNumberEngine32;
	//typedef XORShift64& RandomNumberEngine64;
    typedef pcg32& RandomNumberEngine32;
    typedef pcg64& RandomNumberEngine64;

	inline RandomNumberEngine32 getRandomNumberEngine32Instance() {
		struct Seed {
			Seed() : value(std::random_device{}()) {
				#ifdef CPT_LOG_RANDOM_NUMBER_SEED
					std::cerr << "Seeding global 32-bit random number engine : " << value << std::endl;
				#endif
			}

			uint32_t value;
		};
		thread_local static Seed seed;
		thread_local static std::remove_reference<RandomNumberEngine32>::type engine(seed.value);
		return engine;
	}

	inline RandomNumberEngine64 getRandomNumberEngine64Instance() {
		struct Seed {
			Seed() : value(static_cast<uint64_t>(std::random_device{}()) << 32 | static_cast<uint64_t>(std::random_device{}())) {
				#ifdef CPT_LOG_RANDOM_NUMBER_SEED
					std::cerr << "Seeding global 64-bit random number engine : " << value << std::endl;
				#endif
			}

			uint64_t value;
		};
		thread_local static Seed seed;
		thread_local static std::remove_reference<RandomNumberEngine64>::type engine(seed.value);
		return engine;
	}

	template<class T>
	struct RandomGenerator {
		typedef T ReturnType;

		typedef T result_type;

		virtual ReturnType operator()() = 0;

		virtual ~RandomGenerator() {
		}
	};

	class RandomRealGenerator : public RandomGenerator<double> {
		private:
			typedef std::uniform_real_distribution<> Distribution;

		public:
			/*
				lower bound is inclusive and upper bound is exclusive.
			*/
			RandomRealGenerator(double l, double u)
			: engine(getRandomNumberEngine64Instance())
			, dist(l, u){
			}

			RandomRealGenerator(const RandomRealGenerator&) = default;

			RandomRealGenerator(RandomRealGenerator&&) = default;

			RandomRealGenerator& operator=(const RandomRealGenerator&) = default;

			RandomRealGenerator& operator=(RandomRealGenerator&&) = default;

			ReturnType operator()() {
				return dist(engine);
			}

		private:
			RandomNumberEngine64 engine;
			Distribution dist;
	};

	class RandomIntGenerator : public RandomGenerator<int> {
		private:
			typedef std::uniform_int_distribution<> Distribution;

		public:
			/*
				lower bound is inclusive and upper bound is also inclusive.
			*/
			RandomIntGenerator(int l, int u)
			: engine(getRandomNumberEngine32Instance())
			, dist(l, u) {
			}

			RandomIntGenerator(const RandomIntGenerator&) = default;

			RandomIntGenerator(RandomIntGenerator&&) = default;

			RandomIntGenerator& operator=(const RandomIntGenerator&) = default;

			RandomIntGenerator& operator=(RandomIntGenerator&&) = default;

			ReturnType operator()() {
				return dist(engine);
			}

		private:
			RandomNumberEngine32 engine;
			Distribution dist;
	};

	class RandomLongLongGenerator : public RandomGenerator<long long> {
		private:
			typedef std::uniform_int_distribution<long long> Distribution;

		public:
			/*
				lower bound is inclusive and upper bound is also inclusive.
			*/
			RandomLongLongGenerator(long long l, long long u)
			: engine(getRandomNumberEngine64Instance())
			, dist(l, u) {
			}

			RandomLongLongGenerator(const RandomLongLongGenerator&) = default;

			RandomLongLongGenerator(RandomLongLongGenerator&&) = default;

			RandomLongLongGenerator& operator=(const RandomLongLongGenerator&) = default;

			RandomLongLongGenerator& operator=(RandomLongLongGenerator&&) = default;

			ReturnType operator()() {
				return dist(engine);
			}

		private:
			RandomNumberEngine64 engine;
			Distribution dist;
	};

	class RandomUnsignedIntGenerator : public RandomGenerator<unsigned int> {
		private:
			typedef std::uniform_int_distribution<unsigned int> Distribution;

		public:
			/*
				lower bound is inclusive and upper bound is also inclusive.
			*/
			RandomUnsignedIntGenerator(unsigned int l, unsigned int u)
			: engine(getRandomNumberEngine32Instance())
			, dist(l, u) {
			}

			RandomUnsignedIntGenerator(const RandomUnsignedIntGenerator&) = default;

			RandomUnsignedIntGenerator(RandomUnsignedIntGenerator&&) = default;

			RandomUnsignedIntGenerator& operator=(const RandomUnsignedIntGenerator&) = default;

			RandomUnsignedIntGenerator& operator=(RandomUnsignedIntGenerator&&) = default;

			ReturnType operator()() {
				return dist(engine);
			}

		private:
			RandomNumberEngine32 engine;
			Distribution dist;
	};

	class RandomUnsignedLongLongGenerator : public RandomGenerator<unsigned long long> {
		private:
			typedef std::uniform_int_distribution<unsigned long long> Distribution;

		public:
			/*
				lower bound is inclusive and upper bound is also inclusive.
			*/
			RandomUnsignedLongLongGenerator(unsigned long long l, unsigned long long u)
			: engine(getRandomNumberEngine64Instance())
			, dist(l, u) {
			}

			RandomUnsignedLongLongGenerator(const RandomUnsignedLongLongGenerator&) = default;

			RandomUnsignedLongLongGenerator(RandomUnsignedLongLongGenerator&&) = default;

			RandomUnsignedLongLongGenerator& operator=(const RandomUnsignedLongLongGenerator&) = default;

			RandomUnsignedLongLongGenerator& operator=(RandomUnsignedLongLongGenerator&&) = default;

			ReturnType operator()() {
				return dist(engine);
			}

		private:
			RandomNumberEngine64 engine;
			Distribution dist;
	};

	class RandomPowerLawIntGenerator : public RandomGenerator<int> {
		private:
			typedef std::uniform_real_distribution<> Distribution;

		public:
			/*
				lower bound is inclusive and upper bound is also inclusive.
			*/
			RandomPowerLawIntGenerator(int l, int u, double exponent)
			: l_(l), u_(u), exponent_(exponent)
			, engine(getRandomNumberEngine32Instance())
			, dist(0, 1) {
			}

			RandomPowerLawIntGenerator(const RandomPowerLawIntGenerator&) = default;

			RandomPowerLawIntGenerator(RandomPowerLawIntGenerator&&) = default;

			RandomPowerLawIntGenerator& operator=(const RandomPowerLawIntGenerator&) = default;

			RandomPowerLawIntGenerator& operator=(RandomPowerLawIntGenerator&&) = default;

			/*
				http://mathworld.wolfram.com/RandomNumber.html
				The document explains random 'real' number generator that follows power law.
				We generate random real number that follows power law in the range [l-0.5, u+0.5]
				and rounding to nearest 'integer'.
				Newman (SIAM review, 51, 661) says the rounding method is reliable.
			*/
			ReturnType operator()() {
				double ee = exponent_ + 1;
				return std::pow(((pow(u_ + 0.5, ee) - std::pow(l_ - 0.5, ee)) * dist(engine) + std::pow(l_ - 0.5, ee)), 1 / ee) + 0.5;
			}

		private:
			int l_;
			int u_;
			double exponent_;
			RandomNumberEngine32 engine;
			Distribution dist;
	};

	class RandomPowerLawDoubleGenerator : public RandomGenerator<double> {
		private:
			typedef std::uniform_real_distribution<> Distribution;

		public:
			/*
				lower bound is inclusive and upper bound is also inclusive.
			*/
			RandomPowerLawDoubleGenerator(int l, int u, double exponent)
			: l_(l), u_(u), exponent_(exponent)
			, engine(getRandomNumberEngine32Instance())
			, dist(0, 1) {
			}

			RandomPowerLawDoubleGenerator(const RandomPowerLawDoubleGenerator&) = default;

			RandomPowerLawDoubleGenerator(RandomPowerLawDoubleGenerator&&) = default;

			RandomPowerLawDoubleGenerator& operator=(const RandomPowerLawDoubleGenerator&) = default;

			RandomPowerLawDoubleGenerator& operator=(RandomPowerLawDoubleGenerator&&) = default;

			/*
				http://mathworld.wolfram.com/RandomNumber.html
			*/
			ReturnType operator()() {
				double ee = exponent_ + 1;
				return std::pow(((pow(u_, ee) - std::pow(l_, ee)) * dist(engine) + std::pow(l_, ee)), 1 / ee);
			}

		private:
			int l_;
			int u_;
			double exponent_;
			RandomNumberEngine32 engine;
			Distribution dist;
	};

	class RandomCauchyGenerator : public RandomGenerator<double> {
		private:
			typedef std::cauchy_distribution<> Distribution;

		public:
			RandomCauchyGenerator(double location, double scale)
			: engine(getRandomNumberEngine32Instance())
			, dist(location, scale) {
			}

			RandomCauchyGenerator(const RandomCauchyGenerator&) = default;

			RandomCauchyGenerator(RandomCauchyGenerator&&) = default;

			RandomCauchyGenerator& operator=(const RandomCauchyGenerator&) = default;

			RandomCauchyGenerator& operator=(RandomCauchyGenerator&&) = default;

			ReturnType operator()() {
				return dist(engine);
			}

		private:
			RandomNumberEngine32 engine;
			Distribution dist;
	};

	class RandomGaussianGenerator : public RandomGenerator<double> {
		private:
			typedef std::normal_distribution<> Distribution;

		public:
			RandomGaussianGenerator(double mean, double stddev)
			: engine(getRandomNumberEngine32Instance())
			, dist(mean, stddev) {
			}

			RandomGaussianGenerator(const RandomGaussianGenerator&) = default;

			RandomGaussianGenerator(RandomGaussianGenerator&&) = default;

			RandomGaussianGenerator& operator=(const RandomGaussianGenerator&) = default;

			RandomGaussianGenerator& operator=(RandomGaussianGenerator&&) = default;

			ReturnType operator()() {
				return dist(engine);
			}

		private:
			RandomNumberEngine32 engine;
			Distribution dist;
	};

	class RandomLogNormalGenerator : public RandomGenerator<double> {
		private:
			typedef std::lognormal_distribution<> Distribution;

		public:
			RandomLogNormalGenerator(double location, double scale)
			: engine(getRandomNumberEngine32Instance())
			, dist(location, scale) {
			}

			RandomLogNormalGenerator(const RandomLogNormalGenerator&) = default;

			RandomLogNormalGenerator(RandomLogNormalGenerator&&) = default;

			RandomLogNormalGenerator& operator=(const RandomLogNormalGenerator&) = default;

			RandomLogNormalGenerator& operator=(RandomLogNormalGenerator&&) = default;

			ReturnType operator()() {
				return dist(engine);
			}

		private:
			RandomNumberEngine32 engine;
			Distribution dist;
	};

	class RandomPoissonGenerator : public RandomGenerator<unsigned int> {
		private:
			typedef std::poisson_distribution<> Distribution;

		public:
			RandomPoissonGenerator(double mean)
			: engine(getRandomNumberEngine32Instance())
			, dist(mean) {
			}

			RandomPoissonGenerator(const RandomPoissonGenerator&) = default;

			RandomPoissonGenerator(RandomPoissonGenerator&&) = default;

			RandomPoissonGenerator& operator=(const RandomPoissonGenerator&) = default;

			RandomPoissonGenerator& operator=(RandomPoissonGenerator&&) = default;

			ReturnType operator()() {
				return dist(engine);
			}

		private:
			RandomNumberEngine32 engine;
			Distribution dist;
	};

} //end namespace Cnrc
} //end namespace Snu

#endif
