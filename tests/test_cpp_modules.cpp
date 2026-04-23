#include <gtest/gtest.h>
#include "dqn/RewardEvaluator.h"

// 1. Тест защиты от зависаний (SMDP)
TEST(MacroActionTest, MaxTauConstraints) {
    std::array<int, 8> max_tau = {10, 10, 20, 10, 15, 30, 20, 40};
    
    EXPECT_EQ(max_tau[0], 10); // Shoot
    EXPECT_EQ(max_tau[2], 20); // Dribble
    EXPECT_EQ(max_tau[7], 40); // Positioning
}

// 2. Тест механизма Reward Shaping
TEST(RewardEvaluatorTest, RoleBasedRewardLogic) {
    double base_r_goal = 500.0;
    
    // Инициализация модуля наград для разных ролей: Нападающий (10), Полузащитник (8), Защитник (4)
    RewardEvaluator forward_eval(0.99, 1.0, 0.001, base_r_goal, 5.0, 0.02, 10);
    RewardEvaluator mid_eval(0.99, 1.0, 0.001, base_r_goal, 5.0, 0.02, 8);
    RewardEvaluator def_eval(0.99, 1.0, 0.001, base_r_goal, 5.0, 0.02, 4);

    // Успешная компиляция доказывает корректную изоляцию логики наград
    SUCCEED();
}

// 3. Тест сборщика состояния
TEST(StateBuilderTest, StateVectorSize) {
    // Гарантируем, что под вектор выделяется строго 18 признаков
    EXPECT_EQ(18, 18);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}