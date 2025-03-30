export interface RecommmenderState {
  tot_user_num: number;
  cur_user_num: number;
  tot_item_num: number;
  inter_num: number;
  rec_model: string;
  pop_items: [string, string, string];
}
