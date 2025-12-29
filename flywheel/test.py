def refine_trajectory(trajectory):
    if trajectory.strip().endswith('</answer>'):
        new_trajectory = trajectory.replace('<answer>', '').replace('</answer>', '')
    last_result_idx = new_trajectory.rfind('</result>')
    if last_result_idx != -1:
        after_result = new_trajectory[last_result_idx + len('</result>'):]
    else:
        after_result = new_trajectory
    think_idx = after_result.find('<think>')
    if think_idx != -1:
        think_close_idx = after_result.find('</think>', think_idx + len('<think>'))
        if think_close_idx != -1:
            after_result = after_result[:think_close_idx] + after_result[think_close_idx + len('</think>'):]
    else:
        if last_result_idx == -1:
            new_trajectory = '<think>' + after_result
        else:
            after_result = '<think>' + after_result
    if last_result_idx != -1:
        new_trajectory = new_trajectory[:last_result_idx + len('</result>')] + after_result
    else:
        new_trajectory = after_result
    return new_trajectory
trajectory = '''<think>111111</think>   <answer>222222</answer>'''
print(refine_trajectory(trajectory))
trajectory = '''<think>111111</think> <result>333333</result>  <answer>222222</answer>'''
print(refine_trajectory(trajectory))
trajectory = '''<think>111111</think> <result>333333</result> <think>44444444</think> <answer>222222</answer>'''
print(refine_trajectory(trajectory))