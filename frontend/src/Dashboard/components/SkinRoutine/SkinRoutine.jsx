import React, { useState, useEffect, useCallback } from 'react';
import { Sparkles, Droplets, Calendar, Apple, Lightbulb, Sunrise, Moon, CalendarDays, Sun, Sunset, Pill, DropletIcon, X, Star, Info, CheckCircle2, AlertTriangle, FlaskConical } from 'lucide-react';
import { useUser } from '@clerk/clerk-react';
import './SkinRoutine.css';

const SkinRoutine = ({ latestAnalysis }) => {
  const { user } = useUser();
  const [activeTab, setActiveTab] = useState('routine');
  const [isGenerating, setIsGenerating] = useState(false);
  const [routine, setRoutine] = useState(null);
  const [dietPlan, setDietPlan] = useState(null);
  const [selectedIngredient, setSelectedIngredient] = useState(null);

  const fetchRoutine = useCallback(async () => {
    try {
      setIsGenerating(true);
      const userId = user?.id || localStorage.getItem('last_clerk_user_id') || null;
      if (user?.id) localStorage.setItem('last_clerk_user_id', user.id);
      // Try cached routine first
      const cachedRes = await fetch(`http://localhost:8000/api/skin/routine/${userId}`);
      if (cachedRes.ok) {
        const cached = await cachedRes.json();
        if (cached && cached.routine) {
          const data = cached.routine;
          setRoutine({
            skinType: data.skinType,
            mainConcerns: latestAnalysis?.concerns || [],
            morning: data?.routine?.morning || [],
            evening: data?.routine?.evening || [],
            weekly: data?.routine?.weekly || [],
            ingredients: data?.ingredients || []
          });
          return;
        }
      }
      // If not cached, request generation
      const response = await fetch('http://localhost:8000/api/skin/routine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clerk_user_id: userId })
      });
      if (!response.ok) throw new Error('Failed to generate routine');
      const data = await response.json();
      setRoutine({
        skinType: data.skinType,
        mainConcerns: latestAnalysis?.concerns || [],
        morning: data?.routine?.morning || [],
        evening: data?.routine?.evening || [],
        weekly: data?.routine?.weekly || [],
        ingredients: data?.ingredients || []
      });
    } catch (e) {
      console.error(e);
    } finally {
      setIsGenerating(false);
    }
  }, [latestAnalysis, user]);

  const fetchDiet = useCallback(async (forceNew = false) => {
    try {
      setIsGenerating(true);
      const userId = user?.id || localStorage.getItem('last_clerk_user_id') || null;
      if (user?.id) localStorage.setItem('last_clerk_user_id', user.id);
      if (!forceNew) {
        const cached = await fetch(`http://localhost:8000/api/skin/diet/${userId}`);
        if (cached.ok) {
          const payload = await cached.json();
          if (payload && payload.diet) {
            setDietPlan(payload.diet);
            return;
          }
        }
      }
      const gen = await fetch('http://localhost:8000/api/skin/diet', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ clerk_user_id: userId })
      });
      if (!gen.ok) throw new Error('Failed to generate diet');
      const data = await gen.json();
      setDietPlan(data);
    } catch (e) {
      console.error(e);
    } finally {
      setIsGenerating(false);
    }
  }, [user]);

  const generateDietPlan = async () => {
    await fetchDiet(true);
  };

  useEffect(() => {
    // Capture recent Clerk user id for backend routine generation
    const tryStoreClerk = () => {
      const possible = user?.id || window?.Clerk?.user?.id || null;
      if (possible) localStorage.setItem('last_clerk_user_id', possible);
    };
    tryStoreClerk();
  }, [user]);

  useEffect(() => {
    if (latestAnalysis && !routine) {
      fetchRoutine();
    }
  }, [latestAnalysis, routine, fetchRoutine]);

  useEffect(() => {
    if (activeTab === 'diet' && !dietPlan) {
      fetchDiet(false);
    }
  }, [activeTab, dietPlan, fetchDiet]);

  const IngredientDetails = ({ ingredient, onClose }) => {
    if (!ingredient) return null;
    const titleId = 'ingredient-title';
    return (
      <div className="ingredient-modal" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <div className="ingredient-card" onClick={(e) => e.stopPropagation()}>
          <button className="close-btn" onClick={onClose} aria-label="Close"><X size={18} /></button>
          <div className="ingredient-header">
            <div className="ingredient-icon"><Info size={18} /></div>
            <h3 id={titleId} className="ingredient-title">{ingredient.name}</h3>
          </div>
          {ingredient.benefits && <p className="benefits">{ingredient.benefits}</p>}

          {Array.isArray(ingredient.pros) && ingredient.pros.length > 0 && (
            <div className="ingredient-section">
              <h4 className="section-title"><CheckCircle2 size={16} className="icon" /> Pros</h4>
              <ul>
                {ingredient.pros.map((p, i) => <li key={i}>{p}</li>)}
              </ul>
            </div>
          )}

          {ingredient.usage && (
            <div className="ingredient-section">
              <h4 className="section-title"><FlaskConical size={16} className="icon" /> How to use</h4>
              <p>{ingredient.usage}</p>
            </div>
          )}

          {Array.isArray(ingredient.warnings) && ingredient.warnings.length > 0 && (
            <div className="ingredient-section">
              <h4 className="section-title warning"><AlertTriangle size={16} className="icon" /> Warnings</h4>
              <ul>
                {ingredient.warnings.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            </div>
          )}
        </div>
      </div>
    );
  };

  if (!latestAnalysis) {
    return (
      <div className="skin-routine">
        <div className="empty-state">
          <div className="empty-icon"><Sparkles size={48} /></div>
          <h2>Ready for Your Personalized Routine?</h2>
          <p>Complete your first skin analysis to get AI-powered skincare and diet recommendations tailored just for you!</p>
          <div className="routine-benefits">
            <div className="benefit"><Droplets size={20} className="benefit-icon" /> Customized product recommendations</div>
            <div className="benefit"><Calendar size={20} className="benefit-icon" /> Step-by-step daily routine</div>
            <div className="benefit"><Apple size={20} className="benefit-icon" /> Skin-nourishing diet plan</div>
            <div className="benefit"><Lightbulb size={20} className="benefit-icon" /> Expert tips and guidance</div>
          </div>
        </div>
      </div>
    );
  }

  // Helper to render foods category sections
  const renderFoodsCategory = (label, items = []) => {
    if (!Array.isArray(items) || items.length === 0) return null;
    return (
      <div className="meal-section">
        <h4><Apple size={18} className="inline-icon" /> {label}</h4>
        {items.map((meal, index) => (
          <div key={index} className="meal-item">
            <h5>{meal.food}</h5>
            {meal.benefits && <p>{meal.benefits}</p>}
            <div className="nutrients">
              {(meal.nutrients || []).map((nutrient, idx) => (
                <span key={idx} className="nutrient">{nutrient}</span>
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const foodsToEat = dietPlan?.foodsToEat || {};

  return (
    <div className="skin-routine">
      <div className="routine-header">
        <h2><Sparkles size={24} className="inline-icon" /> Your Personalized Care Plan</h2>
        <p>AI-powered recommendations based on your latest skin analysis</p>
        <div className="header-actions">
          {activeTab === 'routine' ? (
            <button className="generate-btn" onClick={fetchRoutine} disabled={isGenerating}>
              {isGenerating ? 'Generating…' : 'Generate New Routine'}
            </button>
          ) : (
            <button className="generate-btn" onClick={() => fetchDiet(true)} disabled={isGenerating}>
              {isGenerating ? 'Generating…' : 'Generate New Diet Plan'}
            </button>
          )}
        </div>
      </div>

      <div className="tab-selector">
        <button 
          className={`tab-btn ${activeTab === 'routine' ? 'active' : ''}`}
          onClick={() => setActiveTab('routine')}
        >
          <Droplets size={20} className="tab-icon" /> Skincare Routine
        </button>
        <button 
          className={`tab-btn ${activeTab === 'diet' ? 'active' : ''}`}
          onClick={() => setActiveTab('diet')}
        >
          <Apple size={20} className="tab-icon" /> Diet Plan
        </button>
      </div>

      {activeTab === 'routine' && (
        <div className="routine-content">
          {isGenerating && !routine ? (
            <div className="generating-state">
              <div className="loading-animation">
                <div className="loading-circle"></div>
              </div>
              <h3>Creating Your Perfect Routine...</h3>
              <p>Analyzing your skin type and concerns to recommend the best products and steps</p>
            </div>
          ) : routine && (
            <>
              <div className="routine-overview">
                <div className="overview-card">
                  <h3>Routine Summary</h3>
                  <div className="summary-info">
                    <div className="info-item">
                      <span className="info-label">Skin Type:</span>
                      <span className="info-value">{routine.skinType}</span>
                    </div>
                    {Array.isArray(routine.mainConcerns) && routine.mainConcerns.length > 0 && (
                      <div className="info-item">
                        <span className="info-label">Main Concerns:</span>
                        <div className="concerns-list">
                          {routine.mainConcerns.map((concern, index) => (
                            <span key={index} className="concern-tag">{concern}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="routine-schedules">
                <div className="schedule-section">
                  <h3><Sunrise size={20} className="inline-icon" /> Morning Routine</h3>
                  <div className="routine-steps">
                    {routine.morning.map((step) => (
                      <div key={step.step} className="routine-step">
                        <div className="step-number">{step.step}</div>
                        <div className="step-content">
                          <h4>{step.product}</h4>
                          <p>{step.description}</p>
                          <div className="step-details">
                            <span className="duration">{step.duration}</span>
                            <div className="ingredients">
                              {step.ingredients.map((ingredient, index) => (
                                <button key={index} className="ingredient link" onClick={() => setSelectedIngredient(routine.ingredients?.find(i => i.name.toLowerCase() === ingredient.toLowerCase()) || { name: ingredient })}>
                                  <Info size={14} /> {ingredient}
                                </button>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="schedule-section">
                  <h3><Moon size={20} className="inline-icon" /> Evening Routine</h3>
                  <div className="routine-steps">
                    {routine.evening.map((step) => (
                      <div key={step.step} className="routine-step">
                        <div className="step-number">{step.step}</div>
                        <div className="step-content">
                          <h4>{step.product}</h4>
                          <p>{step.description}</p>
                          <div className="step-details">
                            <span className="duration">{step.duration}</span>
                            <div className="ingredients">
                              {step.ingredients.map((ingredient, index) => (
                                <button key={index} className="ingredient link" onClick={() => setSelectedIngredient(routine.ingredients?.find(i => i.name.toLowerCase() === ingredient.toLowerCase()) || { name: ingredient })}>
                                  <Info size={14} /> {ingredient}
                                </button>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="schedule-section">
                  <h3><CalendarDays size={20} className="inline-icon" /> Weekly Treatments</h3>
                  <div className="weekly-treatments">
                    {routine.weekly.map((treatment, index) => (
                      <div key={index} className="treatment-item">
                        <div className="treatment-frequency">{treatment.frequency}</div>
                        <div className="treatment-content">
                          <h4>{treatment.product}</h4>
                          <p>{treatment.description}</p>
                          <div className="ingredients">
                            {treatment.ingredients.map((ingredient, idx) => (
                              <button key={idx} className="ingredient link" onClick={() => setSelectedIngredient(routine.ingredients?.find(i => i.name.toLowerCase() === ingredient.toLowerCase()) || { name: ingredient })}>
                                <Info size={14} /> {ingredient}
                              </button>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="routine-tips">
                  <h3><Lightbulb size={20} className="inline-icon" /> Pro Tips</h3>
                  <ul>
                    <li>Always patch test new products</li>
                    <li>Introduce actives gradually and use sunscreen daily</li>
                    <li>Adjust frequency if irritation occurs</li>
                  </ul>
                </div>
              </div>

              <IngredientDetails ingredient={selectedIngredient} onClose={() => setSelectedIngredient(null)} />
            </>
          )}
        </div>
      )}

      {activeTab === 'diet' && (
        <div className="diet-content">
          {!dietPlan ? (
            <div className="generate-diet">
              <div className="generate-card">
                <h3><Apple size={24} className="inline-icon" /> Generate Your Skin-Nourishing Diet Plan</h3>
                <p>Get personalized nutrition recommendations to support your skin health from the inside out.</p>
                <button 
                  className="generate-btn"
                  onClick={generateDietPlan}
                  disabled={isGenerating}
                >
                  {isGenerating ? 'Generating Plan...' : 'Generate Diet Plan'}
                </button>
              </div>
            </div>
          ) : (
            <div className="diet-plan">
              <div className="diet-overview">
                <h3>Your Personalized Nutrition Plan</h3>
                {Array.isArray(dietPlan.goals) && dietPlan.goals.length > 0 && (
                  <>
                    <p>Targeted nutrition to address:</p>
                    <div className="goals-chips">
                      {dietPlan.goals.map((g, i) => (
                        <span key={i} className="goal-chip">{g}</span>
                      ))}
                    </div>
                  </>
                )}
              </div>

              <div className="daily-meals">
                {renderFoodsCategory('Grains', foodsToEat.grains)}
                {renderFoodsCategory('Vegetables', foodsToEat.vegetables)}
                {renderFoodsCategory('Fruits', foodsToEat.fruits)}
                {renderFoodsCategory('Proteins', foodsToEat.proteins)}
                {renderFoodsCategory('Dals & Legumes', foodsToEat.dals_legumes)}
                {renderFoodsCategory('Spices & Herbs', foodsToEat.spices_herbs)}
                {renderFoodsCategory('Dairy Alternatives', foodsToEat.dairy_alternatives)}
              </div>

              <div className="additional-info">
                <div className="info-section">
                  <h4><Pill size={18} className="inline-icon" /> Recommended Supplements</h4>
                  <div className="supplements">
                    {(dietPlan.supplements || []).map((supplement, index) => (
                      <div key={index} className="supplement-item">
                        <div className="supplement-name">{supplement.name}</div>
                        <div className="supplement-dosage">{supplement.dosage}</div>
                        <div className="supplement-benefits">{supplement.benefits}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="info-section">
                  <h4><DropletIcon size={18} className="inline-icon" /> Hydration Goals</h4>
                  <div className="hydration-info">
                    <div className="water-goal">Target: {dietPlan.hydration?.waterGoal}</div>
                    <ul className="hydration-tips">
                      {(dietPlan.hydration?.tips || []).map((tip, index) => (
                        <li key={index}>{tip}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className="info-section">
                  <h4><X size={18} className="inline-icon" /> Foods to Avoid</h4>
                  <ul className="avoid-list">
                    {(dietPlan.foodsToAvoid || []).map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>

                <div className="info-section">
                  <h4><Star size={18} className="inline-icon" /> Skin Superfoods</h4>
                  <div className="superfoods">
                    {(dietPlan.skinBeneficialFoods || []).map((food, index) => (
                      <div key={index} className="superfood-item">{food}</div>
                    ))}
                  </div>
                </div>

                <div className="info-section">
                  <h4><Lightbulb size={18} className="inline-icon" /> Cooking Tips</h4>
                  <ul className="tips-list">
                    {(dietPlan.cookingTips || []).map((tip, index) => (
                      <li key={index}>{tip}</li>
                    ))}
                  </ul>
                </div>

                
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SkinRoutine;
