import { RecommmenderState } from "./recommender";
import { SocialState } from "./social";

export default interface SystemState {
  recommender: RecommmenderState;
  social: SocialState;
}
